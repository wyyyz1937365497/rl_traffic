"""
测试数据收集 - 验证经验是否正确保存和读取
"""

import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_pickle_files():
    """检查临时目录中的pickle文件"""
    tmp_dir = 'tmp'

    if not os.path.exists(tmp_dir):
        print(f"❌ 临时目录不存在: {tmp_dir}")
        return

    pkl_files = [f for f in os.listdir(tmp_dir) if f.endswith('.pkl')]

    if not pkl_files:
        print(f"❌ 没有找到pickle文件")
        return

    print(f"✅ 找到 {len(pkl_files)} 个pickle文件")

    for pkl_file in pkl_files:
        file_path = os.path.join(tmp_dir, pkl_file)
        print(f"\n{'='*60}")
        print(f"文件: {pkl_file}")
        print(f"大小: {os.path.getsize(file_path) / 1024:.2f} KB")

        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            print(f"✅ 文件加载成功")
            print(f"\n数据结构:")
            for key in data.keys():
                if key == 'experiences':
                    print(f"  - {key}: {len(data[key])} 条经验")
                    if len(data[key]) > 0:
                        print(f"    第一条经验的键: {data[key][0].keys()}")
                        print(f"    junction_id: {data[key][0].get('junction_id')}")
                        print(f"    state shape: {data[key][0].get('state').shape if data[key][0].get('state') is not None else 'None'}")
                        print(f"    reward: {data[key][0].get('reward')}")
                else:
                    print(f"  - {key}: {data[key]}")

        except Exception as e:
            print(f"❌ 加载失败: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_pickle_files()
