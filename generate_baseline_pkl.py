"""
生成 baseline pkl（无控制版本）

复用 generate_submit_bc.py 的数据采集与保存格式，
仅禁用控制指令，确保与提交pkl口径一致。
"""

import argparse

from generate_submit_bc import BCSubmissionGenerator


class BaselineSubmissionGenerator(BCSubmissionGenerator):
    """无控制 baseline 生成器"""

    def _load_bc_model(self):
        """baseline不加载任何模型"""
        print("\n[Baseline] 跳过模型加载（无控制）")
        self.model = None

    def _apply_bc_control(self, step: int, SPEED_LIMIT: float):
        """baseline不施加控制"""
        if step % 300 == 0:
            total_cvs = sum(1 for v in self._safe_vehicle_ids() if self._safe_is_cv(v))
            print(f"  [Baseline] Step {step}: CV车辆={total_cvs}, 本步不施加任何控制")
        return

    @staticmethod
    def _safe_vehicle_ids():
        try:
            import libsumo as traci
        except ImportError:
            import traci
        try:
            return traci.vehicle.getIDList()
        except Exception:
            return []

    @staticmethod
    def _safe_is_cv(veh_id: str):
        try:
            import libsumo as traci
        except ImportError:
            import traci
        try:
            return traci.vehicle.getTypeID(veh_id) == 'CV'
        except Exception:
            return False


def generate_submission_baseline(
    output_path='baseline_submit.pkl',
    sumo_cfg='sumo/sumo.sumocfg',
    device='cpu',
    max_steps=3600,
    seed=42,
):
    """生成无控制 baseline pkl"""
    generator = BaselineSubmissionGenerator(
        sumo_cfg=sumo_cfg,
        checkpoint_path='',
        device=device,
    )
    return generator.generate_pkl(output_path, max_steps, seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='生成baseline pkl（无控制）')
    parser.add_argument('--output', type=str, default='baseline_submit.pkl', help='输出pkl路径')
    parser.add_argument('--sumo-cfg', type=str, default='sumo/sumo.sumocfg', help='SUMO配置路径')
    parser.add_argument('--device', type=str, default='cpu', help='设备（baseline无需GPU）')
    parser.add_argument('--steps', type=int, default=3600, help='仿真步数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    args = parser.parse_args()

    generate_submission_baseline(
        output_path=args.output,
        sumo_cfg=args.sumo_cfg,
        device=args.device,
        max_steps=args.steps,
        seed=args.seed,
    )
