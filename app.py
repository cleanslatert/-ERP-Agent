import streamlit as st
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

# --- 页面配置 ---
st.set_page_config(page_title="工业质检 Agent MVP", page_icon="🏭", layout="wide")
st.title("🏭 工业质检与 ERP 自动化协同 Agent")
st.markdown("基于大模型解决传统产线质检（FQC）及国际贸易订单流转中的数据壁垒问题。")
st.markdown("---")

# --- 侧边栏配置 ---
with st.sidebar:
    st.header("⚙️ 配置中心")
    api_key = st.text_input("OpenAI API Key (必需)", type="password", help="输入你的 OpenAI API Key 即可体验")
    model_name = st.selectbox("选择模型", ["gpt-4o", "gpt-3.5-turbo"])
    quality_threshold = st.slider("合格率阈值 (%)", 80, 100, 98)
    st.markdown("---")
    st.caption("👨‍💻 构建者：Agent Developer")

# --- 核心 Agent 逻辑 ---
class FQCAgent:
    def __init__(self, key):
        self.llm = ChatOpenAI(openai_api_key=key, model=model_name, temperature=0)

    def process(self, raw_text):
        # 1. 定义输出结构
        response_schemas = [
            ResponseSchema(name="model", description="产品型号"),
            ResponseSchema(name="bom_list", description="提取的元器件规格清单, list格式"),
            ResponseSchema(name="defect_summary", description="缺陷情况简述及数量统计"),
            ResponseSchema(name="pass_rate", description="计算出的合格率，纯数字格式(如97.5)"),
            ResponseSchema(name="erp_payload", description="准备存入ERP的结构化JSON数据, 包含订单号、状态等")
        ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()

        # 2. 构建长链推理 Prompt
        prompt = ChatPromptTemplate.from_template(
            "你是一个精通工业自动化与质量控制的专家 Agent。\n"
            "请解析以下包含非标准规格书和质检日志的文本：\n"
            "--- 输入原文 ---\n{input}\n\n"
            "--- 逻辑推理要求 ---\n"
            "1. 解析非标准的规格描述为标准BOM格式。\n"
            "2. 识别显微镜记录中的缺陷总数，并计算合格率：(抽检总数-缺陷总数)/抽检总数*100。\n"
            "3. 将最终结果映射为 ERP 系统可接收的标准格式。\n"
            "按照以下JSON格式输出：\n{format_instructions}"
        )

        _input = prompt.format_prompt(input=raw_text, format_instructions=format_instructions)
        output = self.llm.invoke(_input.to_messages())
        return output_parser.parse(output.content)

# --- 前端展示层 ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📥 步骤 1: 数据输入")
    default_text = """[产品规格]：型号 TX-500, 主控芯片 A1, 贴片电阻 10k, 误差范围 ±1%。
[FQC显微镜记录]：今日产线共抽检 200 件成品。经显微镜检，发现主控芯片引脚虚焊 2 件，外壳划伤 3 件。
[国际贸易信息]：客户-上海森屿行，订单号-PO20260429-001。"""
    
    user_input = st.text_area("粘贴非标准化文本内容（规格书/质检日志）：", value=default_text, height=250)
    run_btn = st.button("🚀 启动 Agent 协作流", use_container_width=True)

if run_btn:
    if not api_key:
        st.error("⚠️ 请先在左侧边栏输入 API Key！")
    else:
        with st.spinner("Agent 正在执行文档解析、长链推理与 ERP 映射..."):
            try:
                agent = FQCAgent(api_key)
                result = agent.process(user_input)

                with col2:
                    st.subheader("📤 步骤 2: 自动化处理结果")
                    
                    st.success(f"**产品型号解析**：{result['model']}")
                    
                    # 自动判断逻辑
                    pass_rate_val = float(result['pass_rate'])
                    is_pass = pass_rate_val >= quality_threshold
                    status_color = "#00cc66" if is_pass else "#ff4b4b"
                    
                    st.markdown(f"**最终合格率判定**：<span style='color:{status_color}; font-size:24px; font-weight:bold'>{pass_rate_val}%</span> " 
                                f"(设定阈值: {quality_threshold}%)", unsafe_allow_html=True)
                    
                    with st.expander("🔍 查看标准 BOM 映射"):
                        st.json(result['bom_list'])
                        
                    with st.expander("🔬 查看缺陷推理分析"):
                        st.write(result['defect_summary'])

                    st.subheader("💾 步骤 3: ERP 数据同步验证")
                    st.json(result['erp_payload'])
                    
                    if is_pass:
                        st.button("✅ 质检达标，确认写入 ERP", type="primary", use_container_width=True)
                    else:
                        st.button("⚠️ 触发异常流转 (生成返修工单)", type="secondary", use_container_width=True)
            
            except Exception as e:
                st.error(f"处理失败，请检查网络或 API Key: {str(e)}")
