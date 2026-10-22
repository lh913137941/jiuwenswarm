# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

"""Rail Extension Manager - 管理用户自定义的 Rail 扩展."""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

from jiuwenswarm.agents.harness.common.plugins.rail_source_policy import (
    ScannedRailFinder,
    ScannedRailLoader,
    contained_extension_dir,
    import_roots,
    read_extension_sources,
    remove_contained_tree,
    resolve_import_source,
    safe_extension_name,
    validate_extension_sources,
)
from jiuwenswarm.common.utils import get_agent_workspace_dir

logger = logging.getLogger(__name__)


def _is_rail_subclass(rail_class: Any) -> bool:
    """Return whether the executed object is a user Rail, not the base itself."""
    if not isinstance(rail_class, type):
        return False
    from openjiuwen.core.single_agent.rail.base import AgentRail
    from openjiuwen.harness.rails import DeepAgentRail

    if rail_class in (DeepAgentRail, AgentRail):
        return False
    return issubclass(rail_class, (DeepAgentRail, AgentRail))


@dataclass
class RailExtension:
    """Rail 扩展信息."""

    name: str  # 扩展名称 (文件夹名称)
    class_name: str = "CustomRail"  # Rail 类名 (从 rail.py 中提取)
    enabled: bool = True  # 是否启用
    description: str = ""  # 描述
    priority: int = 50  # 优先级

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "class_name": self.class_name,
            "enabled": self.enabled,
            "description": self.description,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: dict) -> RailExtension:
        enabled = data.get("enabled", True)
        if not isinstance(enabled, bool):
            enabled = False
        return cls(
            name=data["name"],
            class_name=data.get("class_name", "CustomRail"),
            enabled=enabled,
            description=data.get("description", ""),
            priority=data.get("priority", 50),
        )


class RailManager:
    """Rail 扩展管理器."""

    _instance = None
    _extensions_dir: Path
    _config_file: Path
    _extensions: dict[str, RailExtension] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化 Rail 管理器."""
        if hasattr(self, "_initialized"):
            return

        self._extensions_dir = get_agent_workspace_dir() / "extensions"
        self._config_file = self._extensions_dir / "extensions_config.json"

        # 确保目录存在
        self._extensions_dir.mkdir(parents=True, exist_ok=True)

        # 加载配置
        self._load_config()

        # 跟踪已注册的rail扩展名称
        self._registered_rails: set[str] = set()
        # DeepAgent 实例引用，用于 register/unregister
        self._agent_instance: Any = None
        # 缓存已加载的 rail 实例，确保同一个 rail 只实例化一次
        self._rail_instances: dict[str, Any] = {}
        self._rail_finders: dict[str, ScannedRailFinder] = {}

        self._initialized = True
        logger.info("[RailManager] 初始化完成，扩展目录: %s", self._extensions_dir)

    def _load_config(self) -> None:
        """从配置文件加载扩展信息."""
        if self._config_file.exists():
            try:
                with open(self._config_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if not isinstance(data, dict):
                        raise ValueError("extensions config must be an object")
                    loaded: dict[str, RailExtension] = {}
                    for name, ext_data in data.items():
                        if (
                            not isinstance(ext_data, dict)
                            or ext_data.get("name") != name
                        ):
                            logger.warning("[RailManager] 跳过无效扩展配置: %s", name)
                            continue
                        try:
                            safe_extension_name(name)
                        except ValueError:
                            logger.warning("[RailManager] 跳过非法扩展名: %s", name)
                            continue
                        loaded[name] = RailExtension.from_dict(ext_data)
                    self._extensions = loaded
                logger.info("[RailManager] 加载了 %d 个扩展配置", len(self._extensions))
            except Exception as e:
                logger.error("[RailManager] 加载配置文件失败: %s", e)
                self._extensions = {}
        else:
            self._extensions = {}

    def _save_config(self) -> None:
        """保存扩展信息到配置文件."""
        try:
            data = {name: ext.to_dict() for name, ext in self._extensions.items()}
            with open(self._config_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.debug("[RailManager] 保存配置文件成功")
        except Exception as e:
            logger.error("[RailManager] 保存配置文件失败: %s", e)
            raise

    def list_extensions(self) -> List[dict]:
        """获取所有扩展列表."""
        return [ext.to_dict() for ext in self._extensions.values()]

    def import_extension(self, folder_path: str) -> dict:
        """导入一个新的 Rail 扩展（文件夹结构）.

        Args:
            folder_path: 扩展文件夹路径

        Returns:
            导入的扩展信息

        Raises:
            ValueError: 文件夹名称无效或结构不符合要求
            Exception: 其他错误
        """
        roots = import_roots(self._extensions_dir)
        source_path = resolve_import_source(folder_path, roots)
        name = safe_extension_name(source_path.name)

        # 检查是否已存在
        if name in self._extensions:
            raise ValueError(f"扩展 '{name}' 已存在")

        sources = read_extension_sources(source_path)
        try:
            class_name = validate_extension_sources(sources)
        except ValueError as exc:
            logger.error("[RailManager] rail.py 验证失败: %s", exc)
            raise

        plugin_content = sources["rail.py"]
        dest_path = contained_extension_dir(self._extensions_dir, name)
        try:
            if dest_path.exists() or dest_path.is_symlink():
                remove_contained_tree(dest_path, self._extensions_dir)
            dest_path.mkdir(parents=True, exist_ok=False)
            for filename, text in sources.items():
                target = dest_path / filename
                if target.parent != dest_path:
                    raise ValueError("扩展路径越界")
                target.write_text(text, encoding="utf-8", newline="\n")
            logger.info(
                "[RailManager] 复制文件夹成功: %s -> %s", source_path, dest_path
            )
        except Exception as e:
            logger.error("[RailManager] 复制文件夹失败: %s", e)
            if dest_path.exists() or dest_path.is_symlink():
                remove_contained_tree(dest_path, self._extensions_dir)
            raise

        # 创建扩展记录
        description = self._extract_description(plugin_content)
        priority = self._extract_priority(plugin_content)

        extension = RailExtension(
            name=name,
            class_name=class_name,
            enabled=False,
            description=description,
            priority=priority,
        )

        self._extensions[name] = extension
        self._save_config()

        logger.info("[RailManager] 导入扩展成功: %s", name)
        return extension.to_dict()

    @staticmethod
    def _extract_description(file_str: str) -> str:
        """从文件内容中提取描述信息.

        Args:
            file_str: 文件内容字符串

        Returns:
            提取到的描述
        """
        import re

        # 尝试匹配类文档字符串
        pattern = r'class\s+\w+Rail[^:]*:\s*"""([^"]*?)"""'
        match = re.search(pattern, file_str)
        if match:
            return match.group(1).strip()

        return ""

    @staticmethod
    def _extract_priority(file_str: str) -> int:
        """从文件内容中提取优先级.

        Args:
            file_str: 文件内容字符串

        Returns:
            提取到的优先级
        """
        import re

        # 尝试匹配 priority: int = XX
        pattern = r"priority\s*:\s*int\s*=\s*(\d+)"
        match = re.search(pattern, file_str)
        if match:
            return int(match.group(1))

        return 50  # 默认优先级

    def get_registered_rail_names(self) -> set[str]:
        """获取所有已注册的 rail 扩展名称集合.

        Returns:
            已注册的 rail 名称集合的副本
        """
        return self._registered_rails.copy()

    def delete_extension(self, name: str) -> bool:
        """删除一个扩展（整个文件夹）.

        Args:
            name: 扩展名称

        Returns:
            是否删除成功

        Raises:
            ValueError: 扩展不存在
        """
        safe_extension_name(name)
        if name not in self._extensions:
            raise ValueError(f"扩展 '{name}' 不存在")

        # 如果扩展已注册，从已注册集合中移除
        if name in self._registered_rails:
            self._registered_rails.discard(name)
            logger.info("[RailManager] 扩展 '%s' 从已注册集合中移除", name)

        # 清除缓存的实例
        if name in self._rail_instances:
            del self._rail_instances[name]
            logger.info("[RailManager] 扩展 '%s' 的缓存实例已清除", name)
        self._drop_loaded(name)

        # 删除整个文件夹。只删除扩展目录内的这个名称，不跟随指向外部的链接。
        folder_path = contained_extension_dir(self._extensions_dir, name)
        try:
            remove_contained_tree(folder_path, self._extensions_dir)
        except Exception as e:
            logger.error("[RailManager] 删除文件夹失败: %s", e)
            raise

        # 删除扩展记录
        del self._extensions[name]
        self._save_config()

        logger.info("[RailManager] 删除扩展成功: %s", name)
        return True

    def toggle_extension(self, name: str, enabled: bool) -> dict:
        """切换扩展的启用状态（仅更新配置文件）.

        Args:
            name: 扩展名称
            enabled: 是否启用

        Returns:
            更新后的扩展信息

        Raises:
            ValueError: 扩展不存在
        """
        safe_extension_name(name)
        if not isinstance(enabled, bool):
            raise ValueError("enabled 必须是布尔值")
        if name not in self._extensions:
            raise ValueError(f"扩展 '{name}' 不存在")

        self._extensions[name].enabled = enabled
        self._save_config()

        logger.info("[RailManager] 切换扩展状态（配置文件）: %s -> %s", name, enabled)
        return self._extensions[name].to_dict()

    def set_agent_instance(self, agent_instance: Any) -> None:
        """设置 DeepAgent 实例，用于热更新 rail."""
        self._agent_instance = agent_instance
        logger.info("[RailManager] DeepAgent 实例已设置")

    async def hot_reload_rail(self, name: str, enabled: bool) -> None:
        """热更新 rail：根据 enabled 状态注册或注销 rail 实例.

        Args:
            name: 扩展名称
            enabled: 是否启用

        Raises:
            ValueError: 扩展不存在或未设置 agent 实例
        """
        safe_extension_name(name)
        if not isinstance(enabled, bool):
            raise ValueError("enabled 必须是布尔值")
        if name not in self._extensions:
            raise ValueError(f"扩展 '{name}' 不存在")

        if self._agent_instance is None:
            raise ValueError("DeepAgent 实例未设置，请先调用 set_agent_instance()")

        if enabled:
            # 开启：注册 rail
            if name in self._registered_rails:
                logger.warning("[RailManager] 扩展 '%s' 已注册，跳过", name)
                return

            try:
                rail_instance = self.load_rail_instance_without_enabled_check(name)
                await self._agent_instance.register_rail(rail_instance)
                self._registered_rails.add(name)
                logger.info("[RailManager] 成功注册 rail 扩展: %s", name)
            except Exception as e:
                logger.error("[RailManager] 注册 rail 扩展失败: %s, 错误: %s", name, e)
                raise
        else:
            # 关闭：注销 rail
            if name not in self._registered_rails:
                logger.warning("[RailManager] 扩展 %s 未注册，跳过", name)
                return

            try:
                rail_instance = self.load_rail_instance_without_enabled_check(name)
                await self._agent_instance.unregister_rail(rail_instance)
                self._registered_rails.discard(name)
                logger.info("[RailManager] 成功注销 rail 扩展: %s", name)
            except Exception as e:
                logger.error("[RailManager] 注销 rail 扩展失败: %s, 错误: %s", name, e)
                raise

    def is_rail_registered(self, name: str) -> bool:
        """检查 rail 是否已注册."""
        return name in self._registered_rails

    def get_extensions(self) -> List[dict]:
        """获取所有扩展列表."""
        return [ext.to_dict() for ext in self._extensions.values()]

    def load_rail_instance(self, name: str) -> Any:
        """动态加载并实例化 Rail（需要扩展已启用）.

        Args:
            name: 扩展名称

        Returns:
            Rail 实例

        Raises:
            ValueError: 扩展不存在或未启用
            Exception: 加载失败
        """
        if name not in self._extensions:
            raise ValueError(f"扩展 '{name}' 不存在")

        extension = self._extensions[name]
        if not extension.enabled:
            raise ValueError(f"扩展 '{name}' 未启用")

        return self._load_rail_instance_impl(name)

    def load_rail_instance_without_enabled_check(self, name: str) -> Any:
        """动态加载并实例化 Rail（不检查启用状态，用于热更新）.

        Args:
            name: 扩展名称

        Returns:
            Rail 实例

        Raises:
            ValueError: 扩展不存在
            Exception: 加载失败
        """
        if name not in self._extensions:
            raise ValueError(f"扩展 '{name}' 不存在")

        return self._load_rail_instance_impl(name)

    def _load_rail_class(self, name: str) -> type:
        """加载 Rail 类（不实例化，不缓存）.

        只执行已经通过静态检查的源码副本，并且实例化的对象必须真的是
        DeepAgentRail 或 AgentRail 的子类。配置里的 class_name 不能把
        加载目标改到模块里的其他对象。
        """
        safe_extension_name(name)
        if name not in self._extensions:
            raise ValueError(f"扩展 '{name}' 不存在")

        folder_path = contained_extension_dir(self._extensions_dir, name)
        if not (folder_path / "rail.py").is_file():
            raise ValueError(f"扩展插件文件 '{name}/rail.py' 不存在")

        sources = read_extension_sources(folder_path)
        class_name = validate_extension_sources(sources)
        try:
            module = self._execute_scanned(name, folder_path, sources)
            rail_class = getattr(module, class_name, None)
            if not _is_rail_subclass(rail_class):
                raise ValueError(f"模块中未找到类: {class_name}")
            return rail_class
        except ImportError as e:
            self._drop_loaded(name)
            if "attempted relative import with no known parent package" in str(e):
                raise ValueError(
                    f"扩展 '{name}' 使用了相对导入但缺少 __init__.py 文件。"
                    f"请确保扩展文件夹中包含 __init__.py 文件以支持相对导入。"
                ) from e
            raise
        except Exception as e:
            self._drop_loaded(name)
            logger.error("[RailManager] 加载 Rail 类失败: %s, 错误: %s", name, e)
            raise

    def _execute_scanned(self, name: str, folder: Path, sources: dict[str, str]) -> Any:
        """Execute validated source from memory."""
        self._drop_loaded(name)
        if "__init__.py" in sources:
            package = f"jiuwenswarm_rail_extension_{name}"
            modules: dict[str, tuple[str, str, bool]] = {
                package: (sources["__init__.py"], str(folder / "__init__.py"), True),
            }
            for filename, source in sources.items():
                if filename == "__init__.py":
                    continue
                modules[f"{package}.{Path(filename).stem}"] = (
                    source,
                    str(folder / filename),
                    False,
                )
            finder = ScannedRailFinder(modules)
            sys.meta_path.insert(0, finder)
            self._rail_finders[name] = finder
            importlib.import_module(package)
            return importlib.import_module(f"{package}.rail")

        module_name = f"rail_extension_{name}"
        filename = str(folder / "rail.py")
        loader = ScannedRailLoader(sources["rail.py"], filename)
        spec = importlib.util.spec_from_loader(module_name, loader, origin=filename)
        if spec is None:
            raise ValueError(f"无法加载模块规范: {name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        loader.exec_module(module)
        return module

    def _drop_loaded(self, name: str) -> None:
        finder = self._rail_finders.pop(name, None)
        if finder is not None:
            try:
                sys.meta_path.remove(finder)
            except ValueError:
                pass
        package = f"jiuwenswarm_rail_extension_{name}"
        standalone = f"rail_extension_{name}"
        for key in list(sys.modules):
            if key == standalone or key == package or key.startswith(package + "."):
                sys.modules.pop(key, None)

    def _load_rail_instance_impl(self, name: str) -> Any:
        """加载 rail 实例的实现（缓存机制，确保主 agent 的 rail 只实例化一次）."""
        if name in self._rail_instances:
            logger.debug("[RailManager] 返回缓存的 Rail 实例: %s", name)
            return self._rail_instances[name]

        rail_class = self._load_rail_class(name)
        rail_instance = rail_class()
        self._rail_instances[name] = rail_instance
        logger.info("[RailManager] 加载并缓存 Rail 实例成功: %s", name)
        return rail_instance

    def create_fresh_rail_instance(self, name: str) -> Any:
        """为 team 子 agent 创建独立的 rail 实例（不使用缓存，每次返回新实例）.

        Args:
            name: 扩展名称

        Returns:
            新的 Rail 实例

        Raises:
            ValueError: 扩展不存在
            Exception: 加载失败
        """
        if name not in self._extensions:
            raise ValueError(f"扩展 '{name}' 不存在")

        rail_class = self._load_rail_class(name)
        rail_instance = rail_class()
        logger.debug(
            "[RailManager] 创建新 Rail 实例（team 专用）: %s -> %s", name, rail_instance
        )
        return rail_instance


def get_rail_manager() -> RailManager:
    """获取 Rail 管理器单例."""
    return RailManager()
