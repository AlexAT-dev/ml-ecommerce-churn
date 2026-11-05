import yaml
from src.dynamic_pipeline_builder import DynamicPipelineBuilder


def test_idempotent(config_path):
    """Два послідовні збирання з тією самою конфігурацією дають ідентичну структуру."""
    a = DynamicPipelineBuilder(config_path).build()
    b = DynamicPipelineBuilder(config_path).build()
    assert repr(a) == repr(b), "Збирання не ідемпотентне"


def test_protected_excluded(config_path):
    """Жодна захищена ознака не потрапила до жодної гілки конвеєра."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ct = DynamicPipelineBuilder(config_path).build()
    used = {col for _, _, cols in ct.transformers for col in cols}
    leaked = used & set(cfg.get("protected_attributes", []))
    assert not leaked, f"Захищені ознаки в конвеєрі: {leaked}"


CONFIGS = (
    # ЛР1
    "config/config_lr.yaml", "config/config_rf.yaml",
    # ЛР2: Експеримент 2 (ручні ознаки)
    "config/config_lr_fe.yaml", "config/config_rf_fe.yaml",
    # ЛР2: Експерименти 3–4 — розширений простір після мультимодальної екстракції та DFS
    # (генеруються ноутбуком 02_multimodal_leakage.ipynb)
    "config/generated/config_lr_full.yaml", "config/generated/config_rf_full.yaml",
    "config/generated/config_lr_leaky.yaml", "config/generated/config_rf_leaky.yaml",
    # ЛР3: герметичний конвеєр (безстанові DFS-ознаки + текстовий блок у фолдах)
    "config/generated/config_lr_hermetic.yaml", "config/generated/config_rf_hermetic.yaml",
    # ЛР5: препроцесор сервісу (табличні + клієнтські DFS-ознаки, без відгуків; src/export/export_onnx.py)
    "config/generated/config_rf_serving.yaml",
)


if __name__ == "__main__":
    for path in CONFIGS:
        test_idempotent(path)
        test_protected_excluded(path)
        print(f"{path}: перевірки пройдено")
