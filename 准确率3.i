import pandas as pd

def analyze_survival_predictions(file_path):
    try:
        # 1. 读取 CSV 文件
        df = pd.read_csv(file_path)
        
        # 2. 定义解析函数：将 "1/1", "0/2" 拆解为分子和分母
        def parse_status(status_str):
            try:
                # 确保是字符串并包含分隔符
                if isinstance(status_str, str) and '/' in status_str:
                    numerator, denominator = map(int, status_str.split('/'))
                    return numerator, denominator
            except:
                pass
            return 0, 0  # 异常或空值处理

        # 3. 应用解析，创建临时列
        parsed = df['prediction_status'].apply(parse_status)
        df['correct_num'] = parsed.apply(lambda x: x[0])  # 分子 (预测正确数)
        df['total_denom'] = parsed.apply(lambda x: x[1])  # 分母 (总预测数)

        # --- 统计结果 1: 全局准确率 ---
        # 统计分子和（总预测正确数）与 分母和（总数）
        sum_numerator = df['correct_num'].sum()
        sum_denominator = df['total_denom'].sum()
        
        # --- 统计结果 2: 命中率 (Hit Rate) ---
        # 排除 0/1, 0/2 等完全预测错误的情况，即统计“分子 > 0”的行数
        # 只要有一项预测正确，即视为该条记录“命中”
        hit_rows = df[df['correct_num'] > 0]
        hit_count = len(hit_rows)
        total_rows = len(df)
        hit_rate = hit_count / total_rows if total_rows > 0 else 0

        # --- 输出结果 ---
        print("="*30)
        print("📊 统计报告 (Statistics Report)")
        print("="*30)
        
        print(f"【结果 1：全局统计】")
        print(f"  - 分子和 (总预测正确数): {sum_numerator}")
        print(f"  - 分母和 (预测总数):     {sum_denominator}")
        if sum_denominator > 0:
            print(f"  - 全局准确率:           {sum_numerator / sum_denominator:.2%}")
        
        print("\n" + "-"*30 + "\n")
        
        print(f"【结果 2：命中率统计】")
        print(f"  - 定义: 排除完全错误的行 (即正确数 > 0 的行)")
        print(f"  - 命中行数 (存在正确结果): {hit_count}")
        print(f"  - 总行数 (Total Rows):    {total_rows}")
        print(f"  - 命中率 (Hit Rate):      {hit_rate:.2%}")
        print("="*30)

    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 '{file_path}'，请确认文件路径。")
    except Exception as e:
        print(f"❌ 发生错误: {e}")

# 运行统计
if __name__ == "__main__":
    analyze_survival_predictions('survival_accuracy_report.csv')