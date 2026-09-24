import os  # 导入 os 模块，用于读取环境变量
from langchain_community.vectorstores import FAISS  # 导入 FAISS 向量库，用于存储向量并做相似度检索
from langchain.retrievers import ContextualCompressionRetriever  # 导入上下文压缩检索器，在检索后对文档做压缩/重排
from langchain.retrievers.document_compressors import LLMChainExtractor  # 导入基于 LLM 的文档压缩器，用于抽取相关内容
from langchain_community.embeddings import HuggingFaceBgeEmbeddings  # 导入 BGE 嵌入模型封装，用于把文本向量化
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 导入递归字符切分器，用于把长文档切成块
from langchain_community.document_loaders import TextLoader  # 导入文本加载器，用于读取 txt 文件
from langchain_deepseek import ChatDeepSeek  # 导入 DeepSeek 聊天模型封装

# 导入ColBERT重排器需要的模块
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor  # 导入文档压缩器基类，供自定义重排器继承
from langchain.retrievers.document_compressors import DocumentCompressorPipeline  # 导入压缩管道，用于串联多个压缩/重排步骤
from langchain_core.documents import Document  # 导入 Document 数据结构
from typing import Sequence  # 导入 Sequence 类型，用于函数类型标注
import torch  # 导入 PyTorch，用于张量计算
from transformers import AutoTokenizer, AutoModel  # 导入 HuggingFace 的分词器与模型
import torch.nn.functional as F  # 导入 PyTorch 函数式接口，用于归一化等操作

class ColBERTReranker(BaseDocumentCompressor):  # 定义 ColBERT 重排器类，继承自文档压缩器基类
    """ColBERT重排器"""  # 类的文档字符串

    def __init__(self, **kwargs):  # 构造函数：初始化重排器
        super().__init__(**kwargs)  # 调用父类构造函数完成初始化

        model_name = "bert-base-uncased"  # 指定要加载的预训练模型名称

        # 加载模型和分词器
        object.__setattr__(self, 'tokenizer', AutoTokenizer.from_pretrained(model_name))  # 加载分词器并强制写入属性（绕过 pydantic 校验）
        object.__setattr__(self, 'model', AutoModel.from_pretrained(model_name))  # 加载模型并强制写入属性（绕过 pydantic 校验）
        self.model.eval()  # 将模型切换为评估模式（关闭 dropout 等训练态行为）
        print(f"ColBERT模型加载完成")  # 打印模型加载完成提示

    def encode_text(self, texts):  # 定义文本编码方法：把文本编码成向量
        """ColBERT文本编码"""  # 方法的文档字符串
        inputs = self.tokenizer(  # 用分词器处理输入文本
            texts,  # 待编码的文本
            return_tensors="pt",  # 返回 PyTorch 张量
            padding=True,  # 自动补齐到同一长度
            truncation=True,  # 超过长度时截断
            max_length=128  # 最大长度限制为 128
        )  # 分词结束

        with torch.no_grad():  # 关闭梯度计算，节省显存并加速
            outputs = self.model(**inputs)  # 前向传播得到模型输出

        embeddings = outputs.last_hidden_state  # 取最后一层隐藏状态作为逐 token 的嵌入
        embeddings = F.normalize(embeddings, p=2, dim=-1)  # 在最后一维做 L2 归一化

        return embeddings  # 返回归一化后的嵌入

    def calculate_colbert_similarity(self, query_emb, doc_embs, query_mask, doc_masks):  # 定义 ColBERT 相似度计算方法（MaxSim 操作）
        """ColBERT相似度计算（MaxSim操作）"""  # 方法的文档字符串
        scores = []  # 初始化分数列表

        for i, doc_emb in enumerate(doc_embs):  # 遍历每一个文档的嵌入
            doc_mask = doc_masks[i:i+1]  # 取出当前文档的注意力掩码

            # 计算相似度矩阵
            similarity_matrix = torch.matmul(query_emb, doc_emb.unsqueeze(0).transpose(-2, -1))  # 计算查询 token 与文档 token 的两两内积

            # 应用文档mask
            doc_mask_expanded = doc_mask.unsqueeze(1)  # 扩展掩码维度以对齐相似度矩阵
            similarity_matrix = similarity_matrix.masked_fill(~doc_mask_expanded.bool(), -1e9)  # 把填充位置的相似度置为极小值

            # MaxSim操作
            max_sim_per_query_token = similarity_matrix.max(dim=-1)[0]  # 每个查询 token 取其在所有文档 token 上的最大相似度

            # 应用查询mask
            query_mask_expanded = query_mask.unsqueeze(0)  # 扩展查询掩码维度
            max_sim_per_query_token = max_sim_per_query_token.masked_fill(~query_mask_expanded.bool(), 0)  # 把查询填充位置的分数置 0

            # 求和得到最终分数
            colbert_score = max_sim_per_query_token.sum(dim=-1).item()  # 对所有查询 token 的 MaxSim 求和，得到该文档总分
            scores.append(colbert_score)  # 记录该文档的分数

        return scores  # 返回所有文档的分数列表

    def compress_documents(  # 定义文档压缩方法（此处实际做重排）
        self,  # 实例自身
        documents: Sequence[Document],  # 待处理的文档序列
        query: str,  # 查询文本
        callbacks=None,  # 回调参数（此处未使用）
    ) -> Sequence[Document]:  # 返回处理后的文档序列
        """对文档进行ColBERT重排序"""  # 方法的文档字符串
        if len(documents) == 0:  # 如果文档为空
            return documents  # 直接返回空结果

        # 编码查询
        query_inputs = self.tokenizer(  # 对查询文本进行分词
            [query],  # 查询文本（包装成列表）
            return_tensors="pt",  # 返回 PyTorch 张量
            padding=True,  # 补齐
            truncation=True,  # 截断
            max_length=128  # 最大长度 128
        )  # 分词结束

        with torch.no_grad():  # 关闭梯度计算
            query_outputs = self.model(**query_inputs)  # 前向计算查询的输出
            query_embeddings = F.normalize(query_outputs.last_hidden_state, p=2, dim=-1)  # 对查询 token 嵌入做归一化

        # 编码文档
        doc_texts = [doc.page_content for doc in documents]  # 取出所有文档的正文文本
        doc_inputs = self.tokenizer(  # 对文档文本进行分词
            doc_texts,  # 文档文本列表
            return_tensors="pt",  # 返回 PyTorch 张量
            padding=True,  # 补齐
            truncation=True,  # 截断
            max_length=128  # 最大长度 128
        )  # 分词结束

        with torch.no_grad():  # 关闭梯度计算
            doc_outputs = self.model(**doc_inputs)  # 前向计算文档的输出
            doc_embeddings = F.normalize(doc_outputs.last_hidden_state, p=2, dim=-1)  # 对文档 token 嵌入做归一化

        # 计算ColBERT相似度
        scores = self.calculate_colbert_similarity(  # 调用相似度计算方法
            query_embeddings,  # 查询嵌入
            doc_embeddings,  # 文档嵌入
            query_inputs['attention_mask'],  # 查询的注意力掩码
            doc_inputs['attention_mask']  # 文档的注意力掩码
        )  # 计算结束

        # 排序并返回前5个
        scored_docs = list(zip(documents, scores))  # 把文档与对应分数配对
        scored_docs.sort(key=lambda x: x[1], reverse=True)  # 按分数从高到低排序
        reranked_docs = [doc for doc, _ in scored_docs[:5]]  # 取分数最高的前 5 个文档

        return reranked_docs  # 返回重排后的文档



# 初始化配置
hf_bge_embeddings = HuggingFaceBgeEmbeddings(  # 初始化 BGE 嵌入模型
    model_name="BAAI/bge-large-zh-v1.5"  # 使用中文 large 版 BGE 模型
)  # 初始化结束

llm = ChatDeepSeek(  # 初始化 DeepSeek 聊天模型
    model="deepseek-chat",  # 指定模型名称
    temperature=0.1,  # 温度设为 0.1，略微保留一点随机性
    api_key=os.getenv("DEEPSEEK_API_KEY")  # 从环境变量读取 API Key
)  # 初始化结束

# 1. 加载和处理文档
loader = TextLoader("../../data/C4/txt/ai.txt", encoding="utf-8")  # 创建文本加载器，指定读取 ai.txt（UTF-8 编码）
documents = loader.load()  # 加载文档
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)  # 创建递归字符切分器：块大小 500、重叠 100
docs = text_splitter.split_documents(documents)  # 把文档切分成多个块

# 2. 创建向量存储和基础检索器
vectorstore = FAISS.from_documents(docs, hf_bge_embeddings)  # 用嵌入模型把文档块构建成 FAISS 向量库
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})  # 基于向量库创建基础检索器，召回前 20 条

# 3. 设置ColBERT重排序器
reranker = ColBERTReranker()  # 实例化 ColBERT 重排器

# 4. 设置LLM压缩器
compressor = LLMChainExtractor.from_llm(llm)  # 基于 LLM 创建文档压缩器，用于抽取与问题相关的内容

# 5. 使用DocumentCompressorPipeline组装压缩管道
# 流程: ColBERT重排 -> LLM压缩
pipeline_compressor = DocumentCompressorPipeline(  # 创建文档压缩管道
    transformers=[reranker, compressor]  # 依次执行：先 ColBERT 重排，再 LLM 压缩
)  # 管道创建结束

# 6. 创建最终的压缩检索器
final_retriever = ContextualCompressionRetriever(  # 创建上下文压缩检索器
    base_compressor=pipeline_compressor,  # 指定压缩管道
    base_retriever=base_retriever  # 指定底层基础检索器
)  # 创建结束

# 7. 执行查询并展示结果
query = "AI还有哪些缺陷需要克服？"  # 定义要查询的问题
print(f"\n{'='*20} 开始执行查询 {'='*20}")  # 打印查询开始的分隔线
print(f"查询: {query}\n")  # 打印查询内容

# 7.1 基础检索结果
print(f"--- (1) 基础检索结果 (Top 20) ---")  # 打印小节标题
base_results = base_retriever.get_relevant_documents(query)  # 执行基础检索，取回相关文档
for i, doc in enumerate(base_results):  # 遍历基础检索结果
    print(f"  [{i+1}] {doc.page_content[:100]}...\n")  # 打印每条结果的前 100 个字符（截断显示）

# 7.2 使用管道压缩器的最终结果
print(f"\n--- (2) 管道压缩后结果 (ColBERT重排 + LLM压缩) ---")  # 打印小节标题
final_results = final_retriever.get_relevant_documents(query)  # 执行压缩检索，取回最终文档
for i, doc in enumerate(final_results):  # 遍历最终结果
    print(f"  [{i+1}] {doc.page_content}\n")  # 打印每条结果的完整内容
