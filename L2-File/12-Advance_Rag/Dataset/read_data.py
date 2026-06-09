from datasets import load_dataset
import pandas as pd
import pyarrow.feather as feather

# 直接通过数据集名称加载，datasets 库会自动处理缓存和格式
# 注意：C-MTEB/du_retrieval 可能需要指定配置或拆分，这里假设是默认配置
try:
    dataset = load_dataset("C-MTEB/du_retrieval")
    
    # 查看有哪些拆分 (train, test, queries 等)
    print("可用拆分:", dataset.keys())
    
    # 假设你要的是 'queries' 部分（根据你之前的文件名猜测）
    # 如果数据集结构不同，请调整 key，例如 dataset['test'] 或 dataset['train']
    if 'queries' in dataset:
        df = dataset['queries'].to_pandas()
    elif 'corpus' in dataset:
        df = dataset['corpus'].to_pandas()
    else:
        # 如果只有一个拆分
        df = next(iter(dataset.values())).to_pandas()

    print(df.head())
    
    # 导出 CSV
    df.to_csv("C-MTEB_test_data.csv", index=False, encoding="utf-8-sig")
    print("导出成功！")

except Exception as e: 
    print(f"加载出错: {e}")
    print("尝试方法二...")
    try:
        file_path = r"C:\Users\gs-0033\.cache\huggingface\datasets\C-MTEB___du_retrieval\default\0.0.0\a1a333e290fe30b10f3f56498e3a0d911a693ced\du_retrieval-corpus.arrow"
        # 尝试作为 Feather 文件读取
        table = feather.read_table(file_path)
        df = table.to_pandas()
        print(df.head())
        df.to_csv("C-MTEB_test_data.csv", index=False, encoding="utf-8-sig")
    except Exception as e:
        print(f"Feather 读取失败: {e}")
        # 如果上面失败，尝试作为 IPC 流读取（某些 .arrow 文件其实是 IPC Stream）
        try:
            import pyarrow.ipc as ipc
            with open(file_path, 'rb') as f:
                reader = ipc.open_stream(f)
                table = reader.read_all()
                df = table.to_pandas()
                print(df.head())
                df.to_csv("C-MTEB_test_data.csv", index=False, encoding="utf-8-sig")
        except Exception as e2:
            print(f"IPC 流读取也失败: {e2}")
            print("文件可能已损坏或格式完全不符。建议使用方法一。")


