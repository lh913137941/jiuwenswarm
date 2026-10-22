# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Trust boundary for user Rail extensions.

Rail plugins are in-process subclasses of DeepAgentRail or AgentRail. Loading
one still executes its module body, so the caller must not be able to choose
an arbitrary filesystem path. Import accepts only a direct child of a trusted
root, and every Python file is checked before that module body runs.
"""

from __future__ import annotations

import ast
import importlib.abc
import importlib.util
import os
import re
import stat
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

_MAX_SOURCE_BYTES = 256 * 1024
_MAX_PYTHON_FILES = 32
_MAX_PATH_CHARS = 4096
_SAFE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_RAIL_BASES = frozenset({"DeepAgentRail", "AgentRail"})
_BANNED_CALLS = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "breakpoint",
        "execfile",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "system",
        "popen",
        "spawn",
        "spawnl",
        "spawnle",
        "spawnlp",
        "spawnlpe",
        "spawnv",
        "spawnve",
        "spawnvp",
        "spawnvpe",
        "execl",
        "execle",
        "execlp",
        "execlpe",
        "execv",
        "execve",
        "execvp",
        "execvpe",
        "load_module",
        "exec_module",
        "load_dynamic",
    }
)
_BANNED_MODULES = frozenset(
    {
        "subprocess",
        "ctypes",
        "multiprocessing",
        "pty",
        "commands",
        "pickle",
        "marshal",
        "shelve",
        "importlib",
        "builtins",
        "code",
        "codeop",
        "runpy",
    }
)
_BANNED_ATTRIBUTES = _BANNED_CALLS | frozenset(
    {
        "__builtins__",
        "__globals__",
        "__subclasses__",
        "__bases__",
        "__mro__",
        "gi_frame",
        "gi_code",
        "f_globals",
        "f_locals",
        "f_code",
        "cr_frame",
        "ag_frame",
        "tb_frame",
    }
)


def safe_extension_name(name: str) -> str:
    """Return a single path segment that can be used as an extension id."""
    if not isinstance(name, str) or not _SAFE_NAME.fullmatch(name):
        raise ValueError(f"扩展名称无效: {name!r}"[:200])
    return name


def import_roots(extensions_dir: Path) -> list[Path]:
    """Trusted directories whose direct children may be imported."""
    default = (extensions_dir.parent / "rail_imports").resolve(strict=False)
    default.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(default, 0o700)
    except OSError:
        pass
    roots = [default]
    raw = os.environ.get("JIUWENSWARM_RAIL_IMPORT_ROOTS", "")
    for part in raw.split(os.pathsep):
        text = part.strip()
        if not text:
            continue
        try:
            candidate = Path(text).resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if candidate.is_dir() and not candidate.is_symlink() and candidate not in roots:
            roots.append(candidate)
    return roots


def trusted_root_message(roots: list[Path]) -> str:
    listed = ", ".join(str(root) for root in roots) or "(none)"
    return f"扩展文件夹必须位于受信任导入目录内: {listed}"


def resolve_import_source(folder_path: str, roots: list[Path]) -> Path:
    """Resolve a caller path to a real directory that is a direct child of a root.

    Missing and untrusted paths share the untrusted error until the lexical
    path is inside a root, so callers cannot use this check as an existence
    oracle for the rest of the filesystem.
    """
    if (
        not isinstance(folder_path, str)
        or not folder_path.strip()
        or len(folder_path) > _MAX_PATH_CHARS
        or any(ord(char) < 32 for char in folder_path)
    ):
        raise ValueError(trusted_root_message(roots))
    raw = Path(folder_path)
    if not raw.is_absolute():
        raise ValueError(trusted_root_message(roots))
    try:
        lexical = raw.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise ValueError(trusted_root_message(roots)) from exc
    if not any(path_is_direct_child(lexical, root) for root in roots):
        raise ValueError(trusted_root_message(roots))
    try:
        resolved = raw.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"文件夹不存在或不是目录: {folder_path}") from exc
    if (
        raw.is_symlink()
        or resolved.is_symlink()
        or not any(path_is_direct_child(resolved, root) for root in roots)
    ):
        raise ValueError(trusted_root_message(roots))
    if not resolved.is_dir():
        raise ValueError(f"文件夹不存在或不是目录: {folder_path}")
    try:
        safe_extension_name(resolved.name)
    except ValueError as exc:
        raise ValueError(
            f"文件夹名称 '{resolved.name}' 必须是有效的英文标识符"
        ) from exc
    return resolved


def path_is_direct_child(path: Path, root: Path) -> bool:
    try:
        return path.parent == root and path.is_relative_to(root)
    except (OSError, ValueError):
        return False


def remove_contained_tree(path: Path, root: Path) -> None:
    """Delete an extension directory without following links out of root."""
    try:
        resolved_root = root.resolve(strict=True)
    except (OSError, RuntimeError):
        return
    if path.is_symlink():
        if path.parent.resolve(strict=True) != resolved_root:
            raise ValueError("扩展路径越界")
        path.unlink()
        return
    if not path.exists():
        return
    resolved = path.resolve(strict=True)
    if resolved.parent != resolved_root:
        raise ValueError("扩展路径越界")
    if path.is_file():
        path.unlink()
        return
    _remove_children(path, resolved)
    path.rmdir()


def _remove_children(folder: Path, boundary: Path) -> None:
    for entry in list(folder.iterdir()):
        if entry.is_symlink():
            entry.unlink()
            continue
        if not entry.exists():
            continue
        real = entry.resolve(strict=True)
        if not real.is_relative_to(boundary):
            raise ValueError("扩展路径越界")
        if entry.is_dir():
            _remove_children(entry, boundary)
            entry.rmdir()
        elif entry.is_file():
            entry.unlink()


def contained_extension_dir(extensions_dir: Path, name: str) -> Path:
    """Return the install directory for name, which cannot escape extensions_dir."""
    safe = safe_extension_name(name)
    root = extensions_dir.resolve(strict=False)
    folder = (root / safe).resolve(strict=False)
    if folder.parent != root or not path_is_direct_child(folder, root):
        raise ValueError("扩展名称无效")
    return folder


def read_extension_sources(folder: Path) -> dict[str, str]:
    """Read a flat package of regular Python files without following links."""
    root = folder.resolve(strict=True)
    if folder.is_symlink() or not root.is_dir():
        raise ValueError("扩展目录不能是符号链接")
    sources: dict[str, str] = {}
    for entry in folder.iterdir():
        if entry.is_symlink():
            raise ValueError("扩展目录不能包含符号链接")
        if entry.is_dir():
            if entry.name == "__pycache__":
                continue
            raise ValueError("扩展目录不能包含子目录")
        if not entry.is_file() or entry.suffix != ".py":
            continue
        if entry.name != "__init__.py" and (
            not _SAFE_NAME.fullmatch(entry.stem) or entry.stem.startswith("__")
        ):
            raise ValueError("Python 文件名无效")
        if not entry.name.isascii():
            raise ValueError("Python 文件名无效")
        child = entry.resolve(strict=True)
        if child.parent != root or not child.is_file():
            raise ValueError("扩展路径越界")
        data = read_regular_file(entry, _MAX_SOURCE_BYTES)
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("rail 源文件必须是 UTF-8 文本") from exc
        if "\x00" in text:
            raise ValueError("rail 源文件包含非法字符")
        sources[entry.name] = text
        if len(sources) > _MAX_PYTHON_FILES:
            raise ValueError("扩展 Python 文件过多")
    if "rail.py" not in sources:
        raise ValueError("扩展文件夹必须包含 rail.py 文件")
    return sources


def read_regular_file(path: Path, limit: int) -> bytes:
    """Read a regular file without following a final symlink."""
    flags = os.O_RDONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ValueError("只能读取普通文件") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError("源文件过大或不是普通文件")
        chunks: list[bytes] = []
        remaining = limit
        while remaining > 0:
            blob = os.read(fd, min(remaining, 65536))
            if not blob:
                break
            chunks.append(blob)
            remaining -= len(blob)
        return b"".join(chunks)
    finally:
        os.close(fd)


def validate_extension_sources(sources: dict[str, str]) -> str:
    """Check every file that would be executed and return the Rail class name."""
    class_name = ""
    for filename in sorted(sources):
        try:
            found = validate_source(
                sources[filename],
                require_rail_class=filename == "rail.py",
            )
        except ValueError as exc:
            raise ValueError(f"{filename} 验证失败: {exc}") from exc
        if filename == "rail.py":
            class_name = found
    if not class_name:
        raise ValueError(
            "rail.py 验证失败: 文件必须定义继承自 DeepAgentRail 或 AgentRail 的类"
        )
    return class_name


def validate_source(source: str, *, require_rail_class: bool) -> str:
    """Parse source and reject code that is not a normal Rail module."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError("语法错误") from exc
    _reject_dangerous_nodes(tree)
    class_name = _rail_class_name(tree)
    if require_rail_class and not class_name:
        raise ValueError("文件必须定义继承自 DeepAgentRail 或 AgentRail 的类")
    return class_name


def _rail_class_name(tree: ast.AST) -> str:
    if not isinstance(tree, ast.Module):
        return ""
    found = ""
    defined = False
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and _is_rail_class(node):
            if defined:
                raise ValueError("不能重复定义 Rail 类")
            found = node.name
            defined = True
            continue
        if defined and _binds_name(node, found):
            raise ValueError("不能重绑定 Rail 类")
    return found


def _is_rail_class(node: ast.ClassDef) -> bool:
    names: set[str] = set()
    for base in node.bases:
        if isinstance(base, ast.Name):
            names.add(base.id)
        elif isinstance(base, ast.Attribute):
            names.add(base.attr)
    return bool(names & _RAIL_BASES)


def _binds_name(node: ast.AST, name: str) -> bool:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name == name
    if isinstance(node, ast.Assign):
        return any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        )
    if isinstance(node, ast.AnnAssign):
        return isinstance(node.target, ast.Name) and node.target.id == name
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return any(
            (alias.asname or alias.name.split(".")[-1]) == name for alias in node.names
        )
    return False


def _reject_dangerous_nodes(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in _BANNED_MODULES or alias.name in _BANNED_CALLS:
                    _reject(node)
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root in _BANNED_MODULES:
                _reject(node)
            for alias in node.names:
                if alias.name in _BANNED_CALLS or alias.name in _BANNED_ATTRIBUTES:
                    _reject(node)
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _BANNED_CALLS:
                _reject(node)
            if isinstance(func, ast.Attribute) and func.attr in _BANNED_ATTRIBUTES:
                _reject(node)
        elif isinstance(node, ast.Attribute) and node.attr in _BANNED_ATTRIBUTES:
            _reject(node)
        elif isinstance(node, ast.Name) and node.id in {
            "__builtins__",
            "__loader__",
            "__spec__",
        }:
            _reject(node)


def _reject(node: ast.AST) -> None:
    line = getattr(node, "lineno", "?")
    raise ValueError(f"包含禁止的动态执行或进程调用 (line {line})")


class ScannedRailLoader(importlib.abc.Loader):
    """Execute the source string that already passed validation."""

    def __init__(self, source: str, filename: str) -> None:
        self._source = source
        self._filename = filename

    def create_module(self, spec: Any) -> None:
        return None

    def exec_module(self, module: Any) -> None:
        code = compile(self._source, self._filename, "exec", dont_inherit=True)
        exec(code, module.__dict__)


class ScannedRailFinder(importlib.abc.MetaPathFinder):
    """Serve a scanned package from memory so later imports do not reread disk."""

    def __init__(self, modules: dict[str, tuple[str, str, bool]]) -> None:
        self._modules = modules

    def find_spec(self, fullname: str, path: Any, target: Any = None) -> Any:
        item = self._modules.get(fullname)
        if item is None:
            return None
        source, filename, is_package = item
        loader = ScannedRailLoader(source, filename)
        spec = importlib.util.spec_from_loader(
            fullname,
            loader,
            origin=filename,
            is_package=is_package,
        )
        if spec is not None and is_package:
            # Empty path: sibling imports must hit this finder, not the install directory.
            spec.submodule_search_locations = []
        return spec


def assert_local_extension_peer(transport: Any) -> None:
    """Allow extension mutations from local peers and authenticated desktop transports.

    A browser connecting straight to the agent port sends a foreign Origin and
    is rejected. Named-pipe and stdio peers are strings; those transports are
    authenticated before the connection kernel runs.
    """
    from jiuwenswarm.common.security.web_identity import is_loopback

    remote = getattr(transport, "remote_address", None)
    if isinstance(remote, str) and remote:
        return
    if not isinstance(remote, tuple) or not remote or not is_loopback(remote[0]):
        raise PermissionError("Rail 扩展操作仅允许本机连接")
    origin = _header(transport, "Origin")
    if not origin:
        return
    parsed = urlsplit(origin)
    host = parsed.hostname
    if parsed.scheme not in {"http", "https"} or not host or not is_loopback(host):
        raise PermissionError("Rail 扩展操作仅允许本机连接")


def _header(transport: Any, key: str) -> str | None:
    headers = getattr(transport, "request_headers", None)
    if headers is None:
        headers = getattr(getattr(transport, "request", None), "headers", None)
    getter = getattr(headers, "get", None)
    if not callable(getter):
        return None
    try:
        value = getter(key)
        if value is None:
            value = getter(key.lower())
    except (TypeError, KeyError, ValueError):
        return None
    if value is None:
        return None
    return str(value)
