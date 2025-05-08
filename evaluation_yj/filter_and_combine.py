import os
import pandas as pd
import glob
from typing import Dict, List, Set

# 定义需要保留的文档ID
A_TAG_BART_IDS = {
    'doc_82703374_chunk20', 'doc_7f962bd3_chunk25', 'doc_d96b139b_chunk52',
    'doc_d96b139b_chunk26', 'doc_7f962bd3_chunk7', 'doc_82805800_chunk0',
    'doc_a5d53741_chunk26', 'doc_2ffe3007_chunk9', 'doc_2ffe3007_chunk2',
    'doc_7f962bd3_chunk11', 'doc_82805800_chunk26', 'doc_d0fe9934_chunk3',
    'doc_4f60f16e_chunk2', 'doc_a5d53741_chunk30', 'doc_d96b139b_chunk39',
    'doc_31372ae6_chunk26', 'doc_7f962bd3_chunk8', 'doc_2ffe3007_chunk19',
    'doc_a5d53741_chunk24', 'doc_82703374_chunk22', 'doc_7f962bd3_chunk17',
    'doc_2ffe3007_chunk16', 'doc_82703374_chunk13', 'doc_d96b139b_chunk2',
    'doc_d96b139b_chunk7', 'doc_a5d53741_chunk3', 'doc_d0fe9934_chunk4',
    'doc_82805800_chunk19', 'doc_7f61e8bb_chunk30', 'doc_7f61e8bb_chunk5',
    'doc_d96b139b_chunk65', 'doc_d0fe9934_chunk22', 'doc_a5d53741_chunk8',
    'doc_d96b139b_chunk50'
}

B_TEXT_BART_IDS = {
    'doc_a5d53741_chunk9', 'doc_7f61e8bb_chunk23', 'doc_82805800_chunk19',
    'doc_4f60f16e_chunk8', 'doc_31372ae6_chunk14', 'doc_4f60f16e_chunk4',
    'doc_82805800_chunk12', 'doc_d96b139b_chunk32', 'doc_d96b139b_chunk1',
    'doc_7f962bd3_chunk18', 'doc_2ffe3007_chunk3', 'doc_7f962bd3_chunk14',
    'doc_d96b139b_chunk17', 'doc_2ffe3007_chunk14', 'doc_31372ae6_chunk13',
    'doc_a5d53741_chunk21', 'doc_d96b139b_chunk19', 'doc_a5d53741_chunk1',
    'doc_82805800_chunk24', 'doc_d0fe9934_chunk8', 'doc_82805800_chunk3',
    'doc_7f962bd3_chunk19', 'doc_a5d53741_chunk23', 'doc_7f61e8bb_chunk25',
    'doc_7f61e8bb_chunk6', 'doc_82703374_chunk15', 'doc_2ffe3007_chunk6'
}

A_ONLY_BART_IDS = {
    'doc_82805800_chunk0', 'doc_a5d53741_chunk12', 'doc_d96b139b_chunk3',
    'doc_4f60f16e_chunk5', 'doc_7f61e8bb_chunk6', 'doc_7f962bd3_chunk8',
    'doc_4f60f16e_chunk2', 'doc_a5d53741_chunk6', 'doc_d0fe9934_chunk6',
    'doc_a5d53741_chunk7', 'doc_82805800_chunk2'
}

A_TAG_LED_IDS = {
    'doc_d0fe9934_chunk0', 'doc_82805800_chunk2', 'doc_82805800_chunk4',
    'doc_7f61e8bb_chunk0', 'doc_a5d53741_chunk8', 'doc_82703374_chunk0',
    'doc_2ffe3007_chunk1', 'doc_a5d53741_chunk7', 'doc_82703374_chunk2'
}

# 待添加的ID集合
B_TEXT_LED_IDS = {
    'doc_d96b139b_chunk11', 'doc_2ffe3007_chunk1', 'doc_a5d53741_chunk2', 'doc_4f60f16e_chunk1', 'doc_d96b139b_chunk2', 'doc_d96b139b_chunk0', 'doc_31372ae6_chunk2', 'doc_31372ae6_chunk4'
}
A_ONLY_LED_IDS =  {
    'doc_d96b139b_chunk0', 'doc_82703374_chunk1', 'doc_d0fe9934_chunk0'
}

def get_filter_ids(filename: str) -> Set[str]:
    """
    根据文件名返回对应的过滤ID集合
    """
    if 'A_TAG_BART' in filename:
        return A_TAG_BART_IDS
    elif 'B_TEXT_BART' in filename:
        return B_TEXT_BART_IDS
    elif 'A_ONLY_BART' in filename:
        return A_ONLY_BART_IDS
    elif 'A_TAG_LED' in filename:
        return A_TAG_LED_IDS
    elif 'B_TEXT_LED' in filename:
        return B_TEXT_LED_IDS
    elif 'A_ONLY_LED' in filename:
        return A_ONLY_LED_IDS
    return set()  # 如果没有匹配的过滤规则，返回空集合

def filter_csv(input_file: str, output_dir: str) -> None:
    """
    过滤CSV文件并保存结果
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(input_file)
        original_rows = len(df)

        # 获取过滤ID集合
        filter_ids = get_filter_ids(os.path.basename(input_file))

        if filter_ids:
            # 过滤数据
            df = df[df['doc_id'].isin(filter_ids)]

            # 创建输出文件名
            output_file = os.path.join(output_dir, os.path.basename(input_file))

            # 保存过滤后的文件
            df.to_csv(output_file, index=False)
            print(f"文件: {os.path.basename(input_file)}")
            print(f"  - 原始行数: {original_rows}")
            print(f"  - 过滤后行数: {len(df)}")
            print(f"  - 保留的文档数: {len(df['doc_id'].unique())}")
            print(f"  - 已保存到: {output_file}")
        else:
            print(f"警告: 文件 {input_file} 没有对应的过滤规则")

    except Exception as e:
        print(f"处理文件 {input_file} 时出错: {str(e)}")

def main():
    # 设置输入和输出目录
    input_dir = "generated_summary/Gonzalo"
    output_dir = "generated_summary/GONZALO_filtered"

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有CSV文件
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))

    if not csv_files:
        print(f"在 {input_dir} 中没有找到CSV文件")
        return

    print(f"找到 {len(csv_files)} 个CSV文件需要处理")
    print("开始过滤...\n")

    # 处理每个文件
    for csv_file in csv_files:
        filter_csv(csv_file, output_dir)

    print("\n过滤完成!")
    print(f"过滤后的文件已保存到: {output_dir}")
    print("接下来可以使用 process_gonzalo.py 来处理合并chunk的逻辑")

if __name__ == "__main__":
    main()
