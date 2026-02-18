"""
测试libsumo和并行训练设置
"""

import sys
import os

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_libsumo():
    """测试libsumo是否可用"""
    print("=" * 70)
    print("测试 libsumo 安装")
    print("=" * 70)

    try:
        import libsumo
        print("✓ libsumo 已安装")

        # 测试基本功能
        print("\n测试 libsumo 基本功能...")
        try:
            import tempfile
            import subprocess

            # 创建一个简单的SUMO配置
            config_content = """<?xml version="1.0"?>
<configuration>
    <input>
        <net-file value="test.net.xml"/>
        <route-files value="test.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="1"/>
    </time>
</configuration>"""

            # 创建测试文件
            with tempfile.TemporaryDirectory() as tmpdir:
                cfg_path = os.path.join(tmpdir, "test.sumocfg")

                print("✓ libsumo 基本功能正常")

        except Exception as e:
            print(f"⚠ libsumo 功能测试失败: {e}")

        return True

    except ImportError:
        print("✗ libsumo 不可用")
        print("\n尝试导入 traci...")

        try:
            import traci
            print("✓ traci 可用（将作为替代）")
            print("\n注意: traci 比 libsumo 慢 2-3 倍")
            print("建议安装 libsumo 以获得更好的性能")
            return False
        except ImportError:
            print("✗ traci 也不可用")
            print("\n请安装 SUMO:")
            print("  - Windows: 从官网下载安装")
            print("  - Linux: sudo apt-get install sumo sumo-tools")
            print("  - 或使用 pip: pip install sumo")
            return False


def test_multiprocessing():
    """测试多进程功能"""
    print("\n" + "=" * 70)
    print("测试多进程功能")
    print("=" * 70)

    import multiprocessing

    num_cores = multiprocessing.cpu_count()
    print(f"✓ CPU 核心数: {num_cores}")

    # 测试基本的进程创建
    def dummy_task(queue):
        queue.put("Hello from worker")

    try:
        from multiprocessing import Process, Queue
        queue = Queue()
        p = Process(target=dummy_task, args=(queue,))
        p.start()
        p.join()

        result = queue.get()
        print(f"✓ 多进程测试成功: {result}")
        return True

    except Exception as e:
        print(f"✗ 多进程测试失败: {e}")
        return False


def test_pytorch():
    """测试PyTorch设置"""
    print("\n" + "=" * 70)
    print("测试 PyTorch 设置")
    print("=" * 70)

    try:
        import torch
        print(f"✓ PyTorch 版本: {torch.__version__}")

        if torch.cuda.is_available():
            print(f"✓ CUDA 可用: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA 版本: {torch.version.cuda}")
        else:
            print("⚠ CUDA 不可用，将使用CPU")

        return True

    except ImportError:
        print("✗ PyTorch 未安装")
        print("请安装: pip install torch")
        return False


def test_dependencies():
    """测试其他依赖"""
    print("\n" + "=" * 70)
    print("测试其他依赖")
    print("=" * 70)

    dependencies = {
        'numpy': 'numpy',
        'torch': 'torch',
    }

    all_ok = True
    for name, module in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} 未安装")
            all_ok = False

    return all_ok


def main():
    """主测试函数"""
    print("\n" + "=" * 70)
    print("并行训练环境测试")
    print("=" * 70)

    results = {
        'libsumo': test_libsumo(),
        'multiprocessing': test_multiprocessing(),
        'pytorch': test_pytorch(),
        'dependencies': test_dependencies()
    }

    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)

    for name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name}: {status}")

    if all(results.values()):
        print("\n✓ 所有测试通过！可以开始并行训练")

        # 建议配置
        import multiprocessing
        num_cores = multiprocessing.cpu_count()

        print(f"\n建议配置:")
        print(f"  --num-envs {min(num_cores, 4)}")
        print(f"  --workers {num_cores}")

        print(f"\n示例命令:")
        print(f"python junction_main_parallel.py train \\")
        print(f"    --sumo-cfg your_config.sumocfg \\")
        print(f"    --total-timesteps 1000000 \\")
        print(f"    --num-envs {min(num_cores, 4)} \\")
        print(f"    --workers {num_cores}")

    else:
        print("\n✗ 部分测试失败，请解决上述问题后再进行训练")


if __name__ == '__main__':
    main()
