import os  # 导入 os 模块，用于读取环境变量（API Key）
import asyncio  # 导入 asyncio，用于运行异步评估流程
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings  # 导入向量索引、目录读取器与全局配置 Settings
from llama_index.core.node_parser import SentenceWindowNodeParser, SentenceSplitter  # 导入句子窗口解析器与句子切分器
from llama_index.llms.deepseek import DeepSeek  # 导入 DeepSeek 大模型封装
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # 导入 HuggingFace 嵌入模型封装
from llama_index.core.postprocessor import MetadataReplacementPostProcessor  # 导入元数据替换后处理器（用于句子窗口检索还原上下文）
from llama_index.core.evaluation import (  # 从评估模块导入以下评估组件
    FaithfulnessEvaluator,  # 忠实度评估器（检测幻觉）
    RelevancyEvaluator,  # 相关性评估器（检测是否切题）
    BatchEvalRunner,  # 批量评估运行器（批量、可并行地打分）
)
from llama_index.core.evaluation.eval_utils import get_results_df  # 导入结果转 DataFrame 的工具（便于查看评估明细）
from llama_index.core.evaluation import DatasetGenerator, QueryResponseDataset  # 导入数据集生成器与查询-响应数据集

Settings.llm = DeepSeek(model="deepseek-chat", temperature=0.1, api_key=os.getenv("DEEPSEEK_API_KEY"))  # 全局设置 LLM 为 DeepSeek（作为评估裁判与生成模型）
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")  # 全局设置嵌入模型为 bge-small-en

async def main():  # 定义异步主函数
    # 1. 加载文档
    reader = SimpleDirectoryReader(input_files=["../../data/C3/pdf/IPCC_AR6_WGII_Chapter03.pdf"])  # 创建目录读取器，指定要读取的 PDF 文件
    documents = reader.load_data()  # 加载文档内容为文档对象列表

    # 1.1 加载或生成响应评估数据集
    if os.path.exists("./c6_response_eval_dataset.json"):  # 若本地已存在评估数据集文件
        print("加载响应评估数据集...")  # 打印提示
        response_eval_dataset = QueryResponseDataset.from_json("./c6_response_eval_dataset.json")  # 直接从 JSON 加载，避免重复生成
    else:  # 否则需要生成数据集
        print("生成响应评估数据集...")  # 打印提示
        dataset_generator = DatasetGenerator.from_documents(documents[:10])  # 用前 10 篇文档创建数据集生成器（限制数量以节省时间）
        response_eval_dataset = await dataset_generator.agenerate_dataset_from_nodes(num=15)  # 异步生成 15 个"问题-答案"对
        response_eval_dataset.save_json("./c6_response_eval_dataset.json")  # 保存数据集到本地，方便复用



    # 2. 构建两种不同的RAG查询引擎和检索器进行对比
    # 2.1 句子窗口检索
    sentence_parser = SentenceWindowNodeParser.from_defaults(  # 创建句子窗口解析器（按句切分并附带窗口上下文）
        window_size=3,  # 窗口大小：左右各取 3 句作为上下文
        window_metadata_key="window",  # 把窗口文本存入元数据键 "window"
        original_text_metadata_key="original_text",  # 把原始句子存入元数据键 "original_text"
    )
    sentence_nodes = sentence_parser.get_nodes_from_documents(documents)  # 用该解析器把文档切分成节点
    sentence_index = VectorStoreIndex(sentence_nodes)  # 基于句子窗口节点构建向量索引

    sentence_query_engine = sentence_index.as_query_engine(  # 把索引封装成查询引擎
        similarity_top_k=2,  # 检索时取最相似的 2 个节点
        node_postprocessors=[  # 检索后处理：把命中的句子替换回其窗口上下文
            MetadataReplacementPostProcessor(target_metadata_key="window")  # 用元数据 "window" 的文本替换节点内容
        ],
    )
    sentence_retriever = sentence_index.as_retriever(similarity_top_k=2)  # 基于同一索引创建检索器（取 Top-2）

    # 2.2 常规分块检索（基准）
    base_parser = SentenceSplitter(chunk_size=512)  # 创建常规句子切分器，块大小 512
    base_nodes = base_parser.get_nodes_from_documents(documents)  # 用该切分器把文档切成块节点
    base_index = VectorStoreIndex(base_nodes)  # 基于常规分块节点构建向量索引

    base_query_engine = base_index.as_query_engine(similarity_top_k=2)  # 把基准索引封装成查询引擎（Top-2）
    base_retriever = base_index.as_retriever(similarity_top_k=2)  # 基于基准索引创建检索器（Top-2）

    # 3. 初始化响应评估器
    faithfulness_evaluator = FaithfulnessEvaluator(llm=Settings.llm)  # 初始化忠实度评估器（用 LLM 判断答案是否有据可依）
    relevancy_evaluator = RelevancyEvaluator(llm=Settings.llm)  # 初始化相关性评估器（用 LLM 判断答案是否切题）

    # 4. 执行响应评估对比
    print("开始执行响应评估对比...")  # 打印提示
    evaluators = {"faithfulness": faithfulness_evaluator, "relevancy": relevancy_evaluator}  # 组装评估器字典（多维度）
    queries = response_eval_dataset.queries  # 从数据集中取出所有问题列表

    # 句子窗口检索响应评估
    print("\n=== 评估句子窗口检索 ===")  # 打印小节标题
    sentence_runner = BatchEvalRunner(evaluators, workers=2, show_progress=True)  # 创建批量评估运行器（2 个并行 worker，显示进度）
    sentence_response_results = await sentence_runner.aevaluate_queries(  # 异步批量评估：对每个问题跑引擎并用所有评估器打分
        queries=queries, query_engine=sentence_query_engine  # 传入问题列表与待评估的句子窗口查询引擎
    )

    # 常规分块检索响应评估
    print("\n=== 评估常规分块检索 ===")  # 打印小节标题
    base_runner = BatchEvalRunner(evaluators, workers=2, show_progress=True)  # 创建批量评估运行器（2 个并行 worker）
    base_response_results = await base_runner.aevaluate_queries(  # 异步批量评估基准查询引擎
        queries=queries, query_engine=base_query_engine  # 传入相同问题列表与基准查询引擎
    )

    # 5. 分析并打印对比结果
    print("\n" + "="*60)  # 打印分隔线
    print("响应评估结果对比")  # 打印标题
    print("="*60)  # 打印分隔线

    def calc_response_score(results, metric):  # 定义计算某指标平均得分的辅助函数
        if results and results.get(metric):  # 若结果存在且含该指标
            scores = results[metric]  # 取出该指标的逐条评分列表
            return sum(r.passing for r in scores) / len(scores)  # 用"通过(passing)的比例"作为平均分
        return 0  # 无结果时返回 0

    # 句子窗口检索结果
    sentence_faith = calc_response_score(sentence_response_results, "faithfulness")  # 计算句子窗口检索的忠实度
    sentence_rel = calc_response_score(sentence_response_results, "relevancy")  # 计算句子窗口检索的相关性

    # 常规分块检索结果
    base_faith = calc_response_score(base_response_results, "faithfulness")  # 计算常规分块检索的忠实度
    base_rel = calc_response_score(base_response_results, "relevancy")  # 计算常规分块检索的相关性

    print(f"\n句子窗口检索:")  # 打印句子窗口检索标题
    print(f"  忠实度: {sentence_faith:.1%}")  # 打印忠实度（百分比）
    print(f"  相关性: {sentence_rel:.1%}")  # 打印相关性（百分比）

    print(f"\n常规分块检索:")  # 打印常规分块检索标题
    print(f"  忠实度: {base_faith:.1%}")  # 打印忠实度（百分比）
    print(f"  相关性: {base_rel:.1%}")  # 打印相关性（百分比）



    # 简单对比
    if sentence_faith > base_faith and sentence_rel > base_rel:  # 若两个指标都更高
        print(f"\n✅ 句子窗口检索在两个维度上都优于常规分块检索")  # 打印"全面更优"
    elif sentence_faith > base_faith or sentence_rel > base_rel:  # 若至少一个指标更高
        print(f"\n⚖️  句子窗口检索在某些维度上有优势")  # 打印"部分更优"
    else:  # 两个指标都不更高
        print(f"\n❌ 句子窗口检索未显示明显优势")  # 打印"无明显优势"



if __name__ == "__main__":  # 脚本直接运行时的入口判断
    asyncio.run(main())  # 运行异步主函数
