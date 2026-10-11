"""
Agent测评脚本

跑一组测试问题，输出两类指标：
  健壮性：运行成功率 / 首次执行成功率 / 纠错有效率  —— 只说明「没崩溃」，不等于答对
  正确性：答案准确率                                 —— 与 EXPECTED 真值比对，这才是该宣传的数字
以及分模型的 token 消耗与成本（旧版把 Qwen 的 token 也按 DeepSeek 单价估算，是错的）。
"""
import re
import time

from ecom.analytics.core.graph import build_graph
from ecom.analytics.core.state import make_initial_state
from ecom.analytics.eval.truth import get_eval_set
from ecom.analytics.settings import settings

# 测试问题 + 真值
#   text   -> exec_result 里出现该字符串即算对
#   number -> exec_result 里抽出的数字与该值相对误差在 tol 内即算对
#
# 真值**不写在这里**：由 ecom/analytics/eval/truth.py 从当前数据集的 CSV 算出来，
# 而且用的是标准库 csv 而不是 pandas。写死常数的两个毛病：改数据之后它就过期了，
# 而评测会继续拿旧数字判——它会把错的判成对的，还看不出来。为什么不能用 pandas 算，
# 见那个模块开头的说明（一句话：不能拿被测工具的输出去验证被测工具）。
#
# 题目跟着数据集走：换 ANALYTICS_DATASET，考的就是那份数据的题。
EXPECTED = get_eval_set(settings.DATASET_NAME, settings.RAW_CSV_PATH)
TEST_QUESTIONS = list(EXPECTED.keys())

# 单价（美元/百万 token），请按实际账单核对
PRICES = {
    "deepseek": {"input": 0.27, "output": 1.10},
    # Qwen 单价请按 DashScope 实际价格填入；填 0 表示成本只统计 DeepSeek
    "qwen": {"input": 0.0, "output": 0.0},
}


def judge(exec_result: str, report: str, spec: dict) -> bool:
    """按真值判定这道题答对没有。主看 exec_result（报告是 LLM 转述，可能不出现确切数字）。"""
    result_text = (exec_result or "").replace(",", "")

    if spec["kind"] == "text":
        return spec["value"] in result_text or spec["value"] in (report or "")

    numbers = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", result_text)]
    target = float(spec["value"])
    tol = spec.get("tol", 0.001)
    return any(abs(n - target) <= abs(target) * tol for n in numbers)


def estimate_cost(tokens: dict) -> float:
    """按模型分别计价。tokens 形如 {"deepseek_prompt": .., "deepseek_completion": .., ...}"""
    total = 0.0
    for provider in ("deepseek", "qwen"):
        total += tokens.get(f"{provider}_prompt", 0) / 1_000_000 * PRICES[provider]["input"]
        total += tokens.get(f"{provider}_completion", 0) / 1_000_000 * PRICES[provider]["output"]
    return total


def evaluate():
    app = build_graph()
    results = []

    print("=" * 60)
    print("开始Agent测评...")
    print("=" * 60)

    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n[{i}/{len(TEST_QUESTIONS)}] 问题: {question}")

        start = time.time()
        result = app.invoke(make_initial_state(question))
        elapsed = time.time() - start

        report = result.get("final_report", "")
        exec_result = result.get("exec_result", "")
        retries = result.get("retry_count", 0)
        review_round = result.get("review_round", 0)

        # 健壮性口径：没崩就算过（注意：不代表答案正确）
        robust = "没有结果" not in report and "无法判断" not in report
        # 正确性口径：和真值比对
        correct = judge(exec_result, report, EXPECTED[question])

        tokens = {
            "deepseek_prompt": result.get("deepseek_prompt_tokens", 0),
            "deepseek_completion": result.get("deepseek_completion_tokens", 0),
            "qwen_prompt": result.get("qwen_prompt_tokens", 0),
            "qwen_completion": result.get("qwen_completion_tokens", 0),
        }

        results.append({
            "question": question,
            "robust": robust,
            "correct": correct,
            "first_try": retries == 0,
            "correction_helped": retries > 0 and robust,
            "review_round": review_round,
            "elapsed": round(elapsed, 2),
            "tokens": result.get("total_tokens", 0),
            "token_detail": tokens,
            "retries": retries,
            "report": report,
            "exec_result": exec_result,
        })

        mark = "✅" if correct else "❌"
        print(f"  {mark} 正确={correct} | 耗时:{elapsed:.2f}s | Token:{result.get('total_tokens', 0)} "
              f"| 重试:{retries} | 审查轮次:{review_round}")
        print(f"     结果: {exec_result[:80]}")

    # ===== 汇总 =====
    print("\n" + "=" * 60)
    print("📊 测评结果汇总")
    print("=" * 60)

    total = len(results)
    correct_count = sum(1 for r in results if r["correct"])
    robust_count = sum(1 for r in results if r["robust"])
    first_try_count = sum(1 for r in results if r["first_try"])
    correction_count = sum(1 for r in results if r["correction_helped"])
    avg_time = sum(r["elapsed"] for r in results) / total
    avg_tokens = sum(r["tokens"] for r in results) / total
    avg_retries = sum(r["retries"] for r in results) / total

    ds_prompt = sum(r["token_detail"]["deepseek_prompt"] for r in results)
    ds_completion = sum(r["token_detail"]["deepseek_completion"] for r in results)
    qw_prompt = sum(r["token_detail"]["qwen_prompt"] for r in results)
    qw_completion = sum(r["token_detail"]["qwen_completion"] for r in results)

    avg_cost = estimate_cost({
        "deepseek_prompt": ds_prompt / total,
        "deepseek_completion": ds_completion / total,
        "qwen_prompt": qw_prompt / total,
        "qwen_completion": qw_completion / total,
    })

    print(f"总问题数:         {total}")

    print("\n--- 正确性指标（与真值比对，这才是该看的数字）---")
    print(f"答案准确率:       {correct_count/total*100:.1f}% ({correct_count}/{total})")

    print("\n--- 健壮性指标（只表示没崩溃，不等于答对）---")
    print(f"运行成功率:       {robust_count/total*100:.1f}%")
    print(f"首次执行成功率:   {first_try_count/total*100:.1f}% (DeepSeek第一次写对)")
    print(f"纠错有效率:       {correction_count/total*100:.1f}% (报错后重试改成功)")

    print("\n--- 性能与成本 ---")
    print(f"平均延迟:         {avg_time:.2f}s")
    print(f"平均Token消耗:    {avg_tokens:.0f}")
    print(f"平均重试次数:     {avg_retries:.1f}")
    print(f"Token 明细:       DeepSeek in/out = {ds_prompt}/{ds_completion}, "
          f"Qwen in/out = {qw_prompt}/{qw_completion}")
    print(f"单问题成本估算:   ${avg_cost:.4f} (约¥{avg_cost*7.2:.4f})")
    if PRICES["qwen"]["input"] == 0 and PRICES["qwen"]["output"] == 0:
        print("                  注：Qwen 单价未配置，以上成本只计 DeepSeek")

    print("\n--- 逐题明细 ---")
    for r in results:
        print(f"  {'✅' if r['correct'] else '❌'} {r['question']}  "
              f"期望 vs 实得: {r['exec_result'][:60]}")


if __name__ == "__main__":
    evaluate()
