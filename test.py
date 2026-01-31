import re
import numpy as np
import pandas as pd


def add_weekly_judge_stats(
    input_csv_path: str,
    output_csv_path: str,
    season_col: str = "season",
) -> None:
    """
    对每个 weekX 添加四列：
      1) weekX_judge_score_sum        评委得分总和
      2) weekX_judge_count            评委个数（本周该选手有多少个有效分数）
      3) weekX_judge_score_avg        平均评委得分
      4) weekX_judge_score_avg_rank   本周内平均评委得分排名（默认按 season 内排名，越高越靠前=1）

    规则：
      - 若该选手该周所有评委分数都缺失/为 N/A，则这四列全部填 "N/A"
      - 若部分评委分数缺失，则用有效分数计算 sum/count/avg；排名按 avg 在 season 内做降序排名
    """
    # 读取数据：把 "N/A"/"NA"/空字符串 当成缺失值
    df = pd.read_csv(
        input_csv_path,
        na_values=["N/A", "NA", ""],
        keep_default_na=True,
    )

    # 找出所有 weekX_judgeY_score 这样的列
    score_pat = re.compile(r"^week(\d+)_judge(\d+)_score$", re.IGNORECASE)
    week_to_cols = {}
    for col in df.columns:
        m = score_pat.match(col)
        if m:
            wk = int(m.group(1))
            week_to_cols.setdefault(wk, []).append(col)

    if not week_to_cols:
        raise ValueError("未找到形如 weekX_judgeY_score 的列，请检查列名是否匹配。")

    # 如果没有 season 列，就改为全表排名（不分 season）
    has_season = season_col in df.columns

    out = df.copy()

    for wk, cols in sorted(week_to_cols.items()):
        # 保险：确保是数值，无法转数值的当缺失
        scores = out[cols].apply(pd.to_numeric, errors="coerce")

        # 统计：有效评委个数、总分、平均分
        cnt = scores.count(axis=1)
        s = scores.sum(axis=1, min_count=1)  # 全缺失 => NaN
        avg = s / cnt.replace(0, np.nan)

        # 排名：在 season 内对 avg 降序排名（高分 rank=1）
        if has_season:
            rank = avg.groupby(out[season_col]).rank(method="min", ascending=False)
        else:
            rank = avg.rank(method="min", ascending=False)

        base = f"week{wk}"
        sum_col = f"{base}_judge_score_sum"
        cnt_col = f"{base}_judge_count"
        avg_col = f"{base}_judge_score_avg"
        rank_col = f"{base}_judge_score_avg_rank"

        # “该周为 N/A”：该周所有评委分数都缺失（cnt==0）
        mask_na = cnt == 0

        # 写入列
        out[sum_col] = s
        out[cnt_col] = cnt
        out[avg_col] = avg
        out[rank_col] = rank

        # 按要求：如果为 N/A，则四列都填 "N/A"
        # 这里统一转成 object 以便混用数值和字符串
        out[sum_col] = out[sum_col].astype(object).where(~mask_na, "N/A")
        out[cnt_col] = out[cnt_col].astype(object).where(~mask_na, "N/A")
        out[avg_col] = out[avg_col].astype(object).where(~mask_na, "N/A")

        # rank 转成整数（但保留 N/A）
        out[rank_col] = (
            out[rank_col]
            .round()
            .astype("Int64")      # pandas 可空整数
            .astype(object)
            .where(~mask_na, "N/A")
        )

    # 保存
    out.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"Done. Saved to: {output_csv_path}")


if __name__ == "__main__":
    # 你可以把这里改成你的实际文件路径
    input_path = "/Users/garytchois/Desktop/vs/celebrity_homestate_filled_preserve_NA"
    output_path = "dwts_with_weekly_judge_stats1.csv"

    add_weekly_judge_stats(input_path, output_path)
