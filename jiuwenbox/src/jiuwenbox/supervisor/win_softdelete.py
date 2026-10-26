# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Windows 软删除 DLL 的路径、沙箱 ACL 和挂起进程注入.

DLL 本体在普通 Windows 进程里用 Detours 挂钩删除和创建进程。
Git Bash 不注入；它拉起的 rm 等仍由 DLL 的调试循环装入，装钩与普通进程一样用 DetourAttach。
本模块负责:
- 找到 ``jiuwen_softdelete.dll``
- 给 jbx-sandbox 读和执行权限
- 在 CREATE_SUSPENDED 的 runner 上改导入表，让加载器在主线程起来前映射 DLL

子进程由 DLL 里的 CreateProcessInternalW 钩子继续装, 这里不按每次删除注入.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
from ctypes import wintypes
from pathlib import Path

ENV_ENABLED = "JIUWENBOX_SOFT_DELETE"
ENV_ARCHIVE = "JIUWENBOX_SOFT_DELETE_ARCHIVE"
ENV_DLL = "JIUWENBOX_SOFT_DELETE_DLL"
ENV_RECYCLE_PORT = "JIUWENBOX_SOFT_DELETE_RECYCLE_PORT"
ENV_RECYCLE_TOKEN = "JIUWENBOX_SOFT_DELETE_RECYCLE_TOKEN"
ENV_TRACE = "JIUWENBOX_SOFT_DELETE_TRACE"
_LOG_LAYOUT_KEYS = (
    "JIUWENSWARM_LOG_DIR",
    "JIUWENSWARM_LOG_DATE_ROOT",
    "JIUWENSWARM_DATA_DIR",
)
_DLL_NAME = "jiuwen_softdelete.dll"
_PATCH_ONLY = "JIUWENBOX_SOFT_DELETE_PATCH_ONLY"
_ERROR_ACCESS_DENIED = 5
logger = logging.getLogger(__name__)


def enabled_from_environ() -> bool:
    """只有显式的 ``1`` 才打开. 字符串 true 或其他值都保持关闭."""
    return os.environ.get(ENV_ENABLED) == "1"


def _frozen_dll_candidates(meipass: str) -> list[Path]:
    """冻结包里的查找顺序. 新包在 jiuwenbox_native, 旧包在改名后的目录."""
    root = Path(meipass)
    return [
        root / "jiuwenbox_native" / "native" / _DLL_NAME,
        root / "jiuwenbox_legacy_overlay" / "native" / _DLL_NAME,
    ]


def dll_path() -> Path:
    """返回软删除 DLL 路径。

    环境变量优先。开发布局用本包 ``native`` 目录。
    冻结包不能把 DLL 留在 ``jiuwenbox`` 目录里, 那个目录会盖住 PYZ 里的同名包,
    打包脚本把它放到 ``jiuwenbox_native/native``。
    ``jiuwenbox_legacy_overlay`` 只留给已经把整个目录改名的旧安装包。
    """
    raw = (os.environ.get(ENV_DLL) or "").strip()
    if raw:
        return Path(raw)
    packaged = Path(__file__).resolve().parents[1] / "native" / _DLL_NAME
    if packaged.is_file():
        return packaged
    meipass = getattr(sys, "_MEIPASS", None)
    frozen = _frozen_dll_candidates(meipass) if meipass else []
    for candidate in frozen:
        if candidate.is_file():
            return candidate
    if frozen:
        return frozen[0]
    return packaged


def merge_host_log_layout_env(env: dict[str, str]) -> None:
    """把 box-server 的聚合日志路径写进沙箱 runner/exec 环境.

    jbx-sandbox 的 USERPROFILE 与宿主不同; 若不注入, softdelete DLL 会把
    诊断写到沙箱 profile 下或静默失败, 宿主 full.log 看不到 bash/rm 链路.
    """
    for key in _LOG_LAYOUT_KEYS:
        value = os.environ.get(key)
        if value:
            env.setdefault(key, value)


def merge_soft_delete_env(env: dict[str, str]) -> None:
    """把 runner 里的软删除变量补进子进程环境.

    CreateProcessAsUserW 使用调用方拼出来的环境块, 不会自动继承 runner
    的环境. 缺了归档目录或回收站转发端口, 子进程里的 DLL 就无法按
    box-server 的身份回收文件.
    """
    merge_soft_delete_spawn_env(env)


def merge_soft_delete_spawn_env(env: dict[str, str]) -> None:
    """把当前进程 (box-server 或 runner) 的软删除变量写入目标 env 块.

    runner 第一跳用自定义 env 块启动, 不会自动带上父进程环境; exec 子进程
    同样只拿 IPC 传来的 env + 本函数从 runner os.environ 补齐. 统一入口避免
    只登记了 recycle port 却漏掉 ``JIUWENBOX_SOFT_DELETE`` / DLL 路径.
    """
    merge_host_log_layout_env(env)
    for key in (
        ENV_ENABLED, ENV_ARCHIVE, ENV_DLL, ENV_RECYCLE_PORT, ENV_RECYCLE_TOKEN, ENV_TRACE,
    ):
        value = os.environ.get(key)
        if value:
            env.setdefault(key, value)


def sandbox_can_load_dll(dll: Path, sid: str) -> bool:
    """判断 jbx-sandbox 是否已经能读取并执行这支 DLL.

    打包安装到 Program Files 后, 安装器的 ACL 加固让宿主用户对 DLL 只剩
    ``Users: RX``, 没有 WRITE_DAC; 此时重写 DACL 会被拒绝. 只要现有 ACE
    已经满足加载条件, 就不该再写一次.

    安装器只授 ``Users`` 组, 沙箱用户靠组成员资格拿到权限, 因此判定要把
    本地登录必然携带的组 SID 一并算进去. Deny 优先; 继承专用 ACE 不作用于
    对象本身, 不参与判定.
    """
    try:
        import win32security  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - 仅 Windows 缺 pywin32 时触发
        return False
    from jiuwenbox.supervisor import win_acl, win_constants

    try:
        sid_obj = win32security.ConvertStringSidToSid(sid)
        sd = win32security.GetNamedSecurityInfo(
            str(dll),
            win32security.SE_FILE_OBJECT,
            win32security.DACL_SECURITY_INFORMATION,
        )
    except Exception:  # 读不到 DACL 一律按"不满足"处理, 交给授权流程.
        logger.debug("读取软删除 DLL DACL 失败, 按未授权处理 dll=%s", dll, exc_info=True)
        return False
    dacl = sd.GetSecurityDescriptorDacl()
    if dacl is None:
        return False
    token_keys = {
        win_acl._sid_dedup_key(sid_obj),
        win_constants.SID_BUILTIN_USERS,
        win_constants.SID_AUTHENTICATED_USERS,
        win_constants.SID_EVERYONE,
    }
    inherit_only = getattr(win32security, "INHERIT_ONLY_ACE", 0x8)
    allowed = 0
    denied = 0
    for index in range(dacl.GetAceCount()):
        # GetAce 的返回元组形态随 pywin32 版本变化, 复用 win_acl 的兼容解析.
        ace_type, ace_flags, access_mask, ace_sid = win_acl._parse_getace_tuple(
            dacl.GetAce(index),
        )
        if ace_sid is None or win_acl._sid_dedup_key(ace_sid) not in token_keys:
            continue
        if ace_flags & inherit_only:
            continue
        if ace_type == win_constants.ACCESS_DENIED_ACE_TYPE:
            denied |= int(access_mask)
        elif ace_type == win_constants.ACCESS_ALLOWED_ACE_TYPE:
            allowed |= int(access_mask)
    needed = win_constants.ALLOW_READ_EXECUTE_RIGHTS
    if denied & needed:
        return False
    return (allowed & needed) == needed


def prepare_dll_for_sandbox_user() -> Path:
    """让 jbx-sandbox 能 LoadLibrary 这支 DLL. 失败直接抛出."""
    if sys.platform != "win32":
        raise RuntimeError("soft-delete DLL injection is Windows-only")
    from jiuwenbox.supervisor import win_acl, win_constants, win_setup

    dll = dll_path()
    if not dll.is_file():
        raise FileNotFoundError(f"soft-delete DLL not found: {dll}")
    sid = win_setup.get_sandbox_user_sid()
    if not sid:
        raise RuntimeError("jbx-sandbox SID is missing; cannot grant DLL access")
    if sandbox_can_load_dll(dll, sid):
        logger.debug("软删除 DLL 已对沙箱用户可读, 跳过 DACL 改写 dll=%s", dll)
        # 读权限已经有了, 仍补一次父目录 traverse (best-effort, 内部吞异常):
        # 自定义安装目录下可能只有 DLL 可读而祖先目录不可穿越.
        win_acl.grant_parent_traverse(str(dll), sid)
        return dll
    try:
        win_acl.grant_parent_traverse(str(dll), sid)
        win_acl.grant_ace(
            str(dll),
            sid,
            rights=win_constants.ALLOW_READ_EXECUTE_RIGHTS,
            mode="ALLOW",
            recursive=False,
        )
    except Exception as exc:  # 只有"已够用却被拒"才降级, 其余照常抛出
        if (
            getattr(exc, "winerror", None) != _ERROR_ACCESS_DENIED
            or not sandbox_can_load_dll(dll, sid)
        ):
            raise
        logger.warning(
            "改软删除 DLL DACL 被拒, 但沙箱用户已具备读执行权限, 继续注入 dll=%s: %s",
            dll, exc,
        )
    return dll


def inject_suspended(process_handle: int) -> None:
    """往仍处于挂起状态的进程改导入表，让加载器映射软删除 DLL.

    不恢复主线程. 导入表改失败或目标是 32 位进程时直接返回,
    调用方仍按原方式 ResumeThread.
    """
    if sys.platform != "win32":
        logger.error("软删除注入被跳过: 非 Windows")
        return
    dll = dll_path()
    if not dll.is_file():
        raise FileNotFoundError(f"soft-delete DLL not found: {dll}")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
    kernel.GetModuleHandleW.restype = wintypes.HMODULE
    kernel.LoadLibraryW.argtypes = [wintypes.LPCWSTR]
    kernel.LoadLibraryW.restype = wintypes.HMODULE
    kernel.FreeLibrary.argtypes = [wintypes.HMODULE]
    kernel.FreeLibrary.restype = wintypes.BOOL
    kernel.GetProcAddress.argtypes = [wintypes.HMODULE, ctypes.c_char_p]
    kernel.GetProcAddress.restype = ctypes.c_void_p
    kernel.IsWow64Process.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
    kernel.IsWow64Process.restype = wintypes.BOOL
    process = wintypes.HANDLE(process_handle)
    wow = wintypes.BOOL()
    if kernel.IsWow64Process(process, ctypes.byref(wow)) and wow.value:
        logger.warning(
            "软删除跳过 32 位进程, 进程将不带软删除继续启动 dll=%s", dll,
        )
        return
    already = kernel.GetModuleHandleW(_DLL_NAME)
    loaded_here = None
    if not already:
        os.environ[_PATCH_ONLY] = "1"
        try:
            loaded_here = kernel.LoadLibraryW(str(dll))
        finally:
            os.environ.pop(_PATCH_ONLY, None)
        if not loaded_here:
            err = ctypes.get_last_error()
            logger.error(
                "软删除 DLL 无法在监督进程加载, 子进程将不带软删除启动 winerror=%s dll=%s",
                err, dll,
            )
            raise ctypes.WinError(err)
    module = already or loaded_here
    try:
        addr = kernel.GetProcAddress(module, b"jiuwen_softdelete_patch_process")
        if not addr:
            logger.error("软删除 DLL 缺少 jiuwen_softdelete_patch_process, dll=%s", dll)
            raise RuntimeError("jiuwen_softdelete_patch_process is missing")
        patch_type = ctypes.WINFUNCTYPE(ctypes.c_int, wintypes.HANDLE, use_last_error=True)
        patch = patch_type(addr)
        if not patch(process):
            err = ctypes.get_last_error()
            logger.error(
                "软删除导入表注入失败, 子进程将不带软删除启动 winerror=%s dll=%s",
                err, dll,
            )
            raise RuntimeError(
                f"soft-delete import patch failed (winerror={err}, dll={dll})"
            )
    finally:
        if loaded_here:
            kernel.FreeLibrary(loaded_here)
