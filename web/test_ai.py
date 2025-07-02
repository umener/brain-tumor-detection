#!/usr/bin/env python3
import sys
import os
sys.path.append('.')

# 测试AI分析函数
try:
    from app import get_deepseek_diagnosis
    result = get_deepseek_diagnosis(['meningioma_tumor: 94.05%'])
    print('AI分析测试结果:')
    print(result)
    print('\n结果长度:', len(result))
except Exception as e:
    print('AI分析测试失败:', e)

# 测试记录更新
try:
    import json
    with open('static/records.json', 'r', encoding='utf-8') as f:
        records = json.load(f)
    
    if records:
        first_record = records[0]
        print(f"\n当前最新记录ID: {first_record['record_id']}")
        print(f"AI建议: {repr(first_record['ai_suggestion'])}")
        
        # 模拟更新
        first_record['ai_suggestion'] = "测试AI建议内容"
        
        with open('static/records.json', 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        
        print("测试更新完成")
    else:
        print("没有记录可测试")
        
except Exception as e:
    print('记录更新测试失败:', e)
