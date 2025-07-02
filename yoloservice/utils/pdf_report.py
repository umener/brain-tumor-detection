import os
import subprocess
from datetime import datetime

def generate_pdf_report(result_img, detect_info, llm_advice, save_dir):
    """
    生成包含推理结果图片、检测信息和LLM建议的PDF报告。
    :param result_img: 结果图片路径
    :param detect_info: 检测信息(dict)
    :param llm_advice: LLM建议(str)
    :param save_dir: 保存目录
    :return: PDF文件路径
    """
    # 构造LaTeX内容
    latex = r'''
    \documentclass{article}
    \usepackage{graphicx}
    \usepackage{geometry}
    \geometry{a4paper, margin=1in}
    \begin{document}
    \section*{医学影像智能分析报告}
    \textbf{生成时间:} %s \\
    \section*{检测结果}
    \includegraphics[width=\linewidth]{%s}
    \section*{检测信息}
    \begin{verbatim}
    %s
    \end{verbatim}
    \section*{LLM建议}
    %s
    \end{document}
    ''' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), result_img, str(detect_info), llm_advice)
    tex_path = os.path.join(save_dir, 'report.tex')
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(latex)
    # 调用pdflatex生成PDF
    subprocess.run(['pdflatex', '-output-directory', save_dir, tex_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    pdf_path = os.path.join(save_dir, 'report.pdf')
    return pdf_path
