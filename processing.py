import asyncio, os, json, hashlib, datetime as dt, glob, fitz, httpx, tiktoken, pandas as pd
import re, logging, time
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import difflib
from pathlib import Path
import sqlite3

# ========== 配置区 ==========
API_KEY = os.getenv("SILICON_TOKEN") or "your_api_key_here"
API_ROOT, MODEL = "https://api.siliconflow.cn/v1", "deepseek-ai/DeepSeek-V3"
OUT_DIR, CHUNK_TOK, MAX_OUT = "./output", 45_000, 6_000
ENC = tiktoken.get_encoding("cl100k_base")
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ProcessEntity:
    """工艺实体数据类"""
    text: str
    entity_type: str
    confidence: float = 0.0
    attributes: Dict[str, Any] = None
    start_pos: int = 0
    end_pos: int = 0
    source_file: str = ""
    content_hash: str = ""
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}
        # 生成内容哈希用于去重
        self.content_hash = hashlib.md5(f"{self.text}_{self.entity_type}".encode()).hexdigest()

@dataclass
class ProcessRelation:
    """工艺关系数据类"""
    entity1: str
    entity2: str
    relation_type: str
    confidence: float = 0.0
    attributes: Dict[str, Any] = None
    source_file: str = ""
    content_hash: str = ""
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}
        # 生成内容哈希用于去重
        self.content_hash = hashlib.md5(f"{self.entity1}_{self.entity2}_{self.relation_type}".encode()).hexdigest()

# ========== 优化后的提示词 ==========
ENHANCED_BEARING_PROCESS_PROMPT = """
你是专业的滑动轴承工艺知识提取专家。请从以下工艺文档中精确提取实体和关系信息。

文本内容：
{text}

## 实体提取要求：

### 1. 产品实体 (Product Entities)
- 轴承产品：滑动轴承、径向轴承、止推轴承、复合轴承、轴瓦等
- 零件组件：减摩层、轴承套、推力片、垫片、密封圈、轴承基体等
- 产品特征：内径、外径、厚度、长度、宽度等几何特征
- 技术指标：表面粗糙度、圆柱度、同轴度、跳动等精度指标

### 2. 工艺过程实体 (Process Entities)
- 主工序：粗加工、精加工、热处理、表面处理、装配、检验、离心铸造等
- 具体工步：车削、铣削、磨削、钻孔、铰孔、镗孔、挂锡、熔化、浇注等
- 工艺方法：数控加工、精密磨削、超精研、电镀、阳极氧化等
- 工艺参数：温度、压力、速度、时间、流量等具体数值参数

### 3. 工艺资源实体 (Resource Entities)
- 加工设备：数控车床、磨床、镗床、加工中心、电炉、离心铸造机等
- 工艺装备：夹具、模具、坩埚、中间包、输送泵等
- 检测设备：千分尺、粗糙度仪、扫描电镜、直读光谱仪等
- 辅助设备：加热炉、冷却系统、净化装置等

### 4. 材料实体 (Material Entities)
- 金属材料：巴氏合金、锡基合金、铸铁、钢材、铜合金、铝合金等
- 化学成分：锡(Sn)、锑(Sb)、铜(Cu)、Cu6Sn5、SiC/Si3N等
- 辅助材料：切削液、润滑油、清洗剂、陶瓷涂层、N2气体等
- 材料属性：硬度、强度、熔点、导热性等物理化学性能

### 5. 质量控制实体 (Quality Entities)
- 质量缺陷：黑色斑点、孔洞、夹杂物、裂纹、脱壳、偏析等
- 检测方法：造影技术、扫描电镜、光谱分析、渗透检测等
- 质量标准：公差等级(IT6、IT7)、表面质量(Ra值)、成分要求等
- 控制措施：除气、净化、预热、温度控制等

## 关系提取要求：

### 1. 工艺加工关系
- "工序-零件加工"：某工序加工某零件
- "工步-特征加工"：某工步加工某特征  
- "设备-工序执行"：某设备执行某工序

### 2. 工艺使用关系
- "工序-设备使用"：工序使用设备
- "工序-材料使用"：工序使用材料
- "设备-装备使用"：设备使用工艺装备

### 3. 层次包含关系
- "产品-零件包含"：产品包含零件
- "零件-特征包含"：零件包含特征
- "工艺-工序包含"：工艺包含工序
- "工序-工步包含"：工序包含工步

### 4. 工艺顺序关系
- "工序-前后顺序"：工序之间的先后关系
- "工步-执行顺序"：工步之间的执行顺序

### 5. 质量控制关系
- "缺陷-零件位置"：缺陷出现在某零件上
- "检测-缺陷发现"：检测方法发现缺陷
- "措施-缺陷消除"：控制措施消除缺陷

### 6. 参数控制关系
- "工序-参数控制"：工序控制的参数
- "设备-参数设定"：设备的参数设定
- "材料-性能参数"：材料的性能参数

## 输出格式要求：
1. 严格按照JSON格式输出
2. 数值参数必须提取具体数值和单位
3. 置信度基于文本明确程度(0.5-1.0)
4. 属性字段包含详细分类信息

输出JSON：
{{
  "entities": [
    {{
      "text": "实体文本",
      "type": "实体类型",
      "start": 起始位置,
      "end": 结束位置,
      "confidence": 置信度,
      "attributes": {{
        "category": "详细分类",
        "unit": "单位(如适用)",
        "value": "数值(如适用)",
        "specification": "规格说明(如适用)"
      }}
    }}
  ],
  "relations": [
    {{
      "entity1": "实体1",
      "entity2": "实体2", 
      "type": "关系类型",
      "confidence": 置信度,
      "attributes": {{
        "direction": "关系方向",
        "strength": "关系强度",
        "context": "上下文说明"
      }}
    }}
  ]
}}
"""

class ProcessedFileTracker:
    """已处理文件跟踪器"""
    
    def __init__(self, db_path: str = "./output/processed_files.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """初始化数据库"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS processed_files (
                file_path TEXT PRIMARY KEY,
                file_hash TEXT,
                process_time TEXT,
                entity_count INTEGER,
                relation_count INTEGER
            )
        ''')
        conn.commit()
        conn.close()
    
    def is_processed(self, file_path: str, file_hash: str) -> bool:
        """检查文件是否已处理"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            'SELECT file_hash FROM processed_files WHERE file_path = ?',
            (file_path,)
        )
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0] == file_hash:
            return True
        return False
    
    def mark_processed(self, file_path: str, file_hash: str, 
                      entity_count: int, relation_count: int):
        """标记文件为已处理"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            INSERT OR REPLACE INTO processed_files 
            (file_path, file_hash, process_time, entity_count, relation_count)
            VALUES (?, ?, ?, ?, ?)
        ''', (file_path, file_hash, dt.datetime.now().isoformat(), 
              entity_count, relation_count))
        conn.commit()
        conn.close()
    
    def get_processed_files(self) -> List[Dict[str, Any]]:
        """获取已处理文件列表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute('SELECT * FROM processed_files')
        files = [dict(zip([col[0] for col in cursor.description], row)) 
                for row in cursor.fetchall()]
        conn.close()
        return files

class BearingProcessExtractor:
    """增强版滑动轴承工艺知识提取器"""
    
    def __init__(self):
        self.entities_cache = defaultdict(set)  # 使用set避免重复
        self.relations_cache = defaultdict(set)
        self.standard_terms = self._load_standard_terms()
        self.file_tracker = ProcessedFileTracker()
        
    def _load_standard_terms(self) -> Dict[str, List[str]]:
        """加载标准术语库"""
        return {
            "工序": ["粗加工", "半精加工", "精加工", "超精加工", "热处理", 
                    "表面处理", "装配", "检验", "离心铸造", "挂锡", "熔化"],
            "设备": ["数控车床", "磨床", "镗床", "加工中心", "电炉", "离心铸造机",
                    "清洗机", "加热炉", "中间保温包", "电动锡输送泵"],
            "材料": ["巴氏合金", "锡基合金", "铸铁", "钢材", "铜合金", "铝合金",
                    "锡", "锑", "铜", "陶瓷涂层"],
            "缺陷": ["黑色斑点", "孔洞", "夹杂物", "裂纹", "脱壳", "偏析",
                    "结合强度低", "结晶粗大"],
            "参数": ["温度", "压力", "速度", "时间", "流量", "转速", "进给量"]
        }
    
    def get_file_hash(self, file_path: str) -> str:
        """计算文件哈希值"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def extract_numerical_parameters(self, text: str) -> List[ProcessEntity]:
        """提取数值参数"""
        entities = []
        
        # 温度参数
        temp_pattern = r'(\d+(?:\.\d+)?)\s*[℃°C]'
        for match in re.finditer(temp_pattern, text):
            entities.append(ProcessEntity(
                text=match.group(0),
                entity_type="工艺过程实体",
                confidence=0.95,
                attributes={
                    "category": "工艺参数",
                    "parameter_type": "温度",
                    "unit": "℃",
                    "value": match.group(1)
                }
            ))
        
        # 尺寸参数
        size_pattern = r'(\d+(?:\.\d+)?)\s*mm'
        for match in re.finditer(size_pattern, text):
            entities.append(ProcessEntity(
                text=match.group(0),
                entity_type="产品实体",
                confidence=0.9,
                attributes={
                    "category": "尺寸参数",
                    "unit": "mm",
                    "value": match.group(1)
                }
            ))
        
        # 压力参数
        pressure_pattern = r'(\d+(?:\.\d+)?)\s*[Mm]?[Pp]a'
        for match in re.finditer(pressure_pattern, text):
            entities.append(ProcessEntity(
                text=match.group(0),
                entity_type="工艺过程实体",
                confidence=0.9,
                attributes={
                    "category": "工艺参数",
                    "parameter_type": "压力",
                    "unit": "MPa",
                    "value": match.group(1)
                }
            ))
        
        return entities
    
    def extract_process_sequences(self, text: str) -> List[ProcessRelation]:
        """提取工艺序列关系"""
        relations = []
        
        # 工艺流程箭头关系
        arrow_pattern = r'([^→\n]+)→([^→\n]+)'
        for match in re.finditer(arrow_pattern, text):
            step1 = match.group(1).strip()
            step2 = match.group(2).strip()
            
            relations.append(ProcessRelation(
                entity1=step1,
                entity2=step2,
                relation_type="工序-前后顺序",
                confidence=0.9,
                attributes={
                    "direction": "前后",
                    "strength": "强",
                    "context": "工艺流程"
                }
            ))
        
        return relations

def calculate_file_hash(file_path: str) -> str:
    """计算文件内容哈希"""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def split_text_intelligently(text: str, limit: int = CHUNK_TOK) -> List[str]:
    """智能文本分割，保持语义完整性"""
    # 首先按段落分割
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        # 检查当前段落加入后是否超出限制
        test_chunk = current_chunk + "\n\n" + para if current_chunk else para
        if len(ENC.encode(test_chunk)) <= limit:
            current_chunk = test_chunk
        else:
            # 如果当前块不为空，先保存
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = para
            else:
                # 单个段落过长，需要进一步分割
                sentences = re.split(r'[。！？；\n]', para)
                temp_chunk = ""
                for sent in sentences:
                    test_sent = temp_chunk + sent if temp_chunk else sent
                    if len(ENC.encode(test_sent)) <= limit:
                        temp_chunk = test_sent
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk)
                        temp_chunk = sent
                if temp_chunk:
                    current_chunk = temp_chunk
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def extract_text_enhanced(pdf_path: str) -> str:
    """增强的PDF文本提取"""
    try:
        with fitz.open(pdf_path) as doc:
            text_blocks = []
            for page in doc:
                # 获取文本块，保持结构
                blocks = page.get_text("dict")["blocks"]
                page_text = ""
                for block in blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            line_text = ""
                            for span in line["spans"]:
                                line_text += span["text"]
                            page_text += line_text + "\n"
                text_blocks.append(page_text)
            
            # 清理和标准化文本
            full_text = "\n\n".join(text_blocks)
            # 移除多余空白
            full_text = re.sub(r'\n\s*\n', '\n\n', full_text)
            full_text = re.sub(r' +', ' ', full_text)
            
            return full_text
    except Exception as e:
        logger.error(f"PDF提取失败 {pdf_path}: {e}")
        return ""

async def call_llm_with_retry(chunk: str, client: httpx.AsyncClient, 
                             prompt_template: str = ENHANCED_BEARING_PROCESS_PROMPT,
                             max_retries: int = 3) -> dict:
    """带重试机制的LLM调用"""
    for attempt in range(max_retries):
        try:
            req = {
                "model": MODEL,
                "stream": True,
                "max_tokens": MAX_OUT,
                "temperature": 0.05,  # 降低温度提高一致性
                "messages": [
                    {"role": "system", "content": "你是专业的工艺知识提取专家。严格按照要求输出JSON格式，确保格式正确完整。"},
                    {"role": "user", "content": prompt_template.format(text=chunk)}
                ]
            }
            
            async with client.stream("POST", f"{API_ROOT}/chat/completions",
                                   headers=HEADERS, json=req, timeout=180) as r:
                r.raise_for_status()
                pieces = []
                async for raw in r.aiter_lines():
                    if not raw.startswith("data: "):
                        continue
                    seg = raw[6:].strip()
                    if seg in ("", "[DONE]"):
                        continue
                    try:
                        delta = json.loads(seg)["choices"][0]["delta"]
                        pieces.append(delta.get("content", ""))
                    except (json.JSONDecodeError, KeyError):
                        continue
                
                buf = "".join(pieces).strip()
                if not buf:
                    continue
                
                # 尝试解析JSON
                try:
                    return json.loads(buf)
                except json.JSONDecodeError:
                    # 尝试修复JSON
                    start, end = buf.find("{"), buf.rfind("}")
                    if start != -1 and end != -1:
                        try:
                            return json.loads(buf[start:end+1])
                        except:
                            pass
                    
                    # 如果是最后一次尝试，返回空结果
                    if attempt == max_retries - 1:
                        logger.error(f"JSON解析失败: {buf[:200]}")
                        return {}
                    
        except Exception as e:
            logger.warning(f"LLM调用失败 (尝试 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # 指数退避
            
    return {}

def load_existing_data() -> Tuple[Set[str], Set[str]]:
    """加载现有数据，返回已存在的实体和关系哈希集合"""
    entity_hashes = set()
    relation_hashes = set()
    
    entity_file = os.path.join(OUT_DIR, "bearing_process_entities.csv")
    relation_file = os.path.join(OUT_DIR, "bearing_process_relations.csv")
    
    if os.path.exists(entity_file):
        df = pd.read_csv(entity_file)
        entity_hashes = set(df.get('content_hash', []))
        logger.info(f"加载现有实体 {len(entity_hashes)} 条")
    
    if os.path.exists(relation_file):
        df = pd.read_csv(relation_file)
        relation_hashes = set(df.get('content_hash', []))
        logger.info(f"加载现有关系 {len(relation_hashes)} 条")
    
    return entity_hashes, relation_hashes

def merge_and_deduplicate_advanced(all_entities: List[ProcessEntity], 
                                 all_relations: List[ProcessRelation],
                                 existing_entity_hashes: Set[str],
                                 existing_relation_hashes: Set[str]) -> Tuple[List[ProcessEntity], List[ProcessRelation]]:
    """高级合并和去重"""
    
    # 实体去重 - 基于内容哈希和语义相似度
    unique_entities = {}
    for entity in all_entities:
        # 跳过已存在的实体
        if entity.content_hash in existing_entity_hashes:
            continue
            
        # 语义去重
        key = f"{entity.text.lower()}_{entity.entity_type}"
        if key not in unique_entities or entity.confidence > unique_entities[key].confidence:
            unique_entities[key] = entity
    
    # 关系去重
    unique_relations = {}
    for relation in all_relations:
        # 跳过已存在的关系
        if relation.content_hash in existing_relation_hashes:
            continue
            
        # 语义去重
        key = f"{relation.entity1.lower()}_{relation.entity2.lower()}_{relation.relation_type}"
        if key not in unique_relations or relation.confidence > unique_relations[key].confidence:
            unique_relations[key] = relation
    
    return list(unique_entities.values()), list(unique_relations.values())

def append_to_csv(entities: List[ProcessEntity], relations: List[ProcessRelation]):
    """追加数据到CSV文件"""
    
    if entities:
        # 准备实体数据
        entity_data = []
        for entity in entities:
            entity_data.append({
                "entity_id": f"E_{entity.content_hash[:8]}",
                "entity_text": entity.text,
                "entity_type": entity.entity_type,
                "confidence": entity.confidence,
                "attributes": json.dumps(entity.attributes, ensure_ascii=False),
                "source_file": entity.source_file,
                "content_hash": entity.content_hash,
                "create_time": dt.datetime.now().isoformat()
            })
        
        # 追加到CSV
        entity_file = os.path.join(OUT_DIR, "bearing_process_entities.csv")
        df_new = pd.DataFrame(entity_data)
        
        if os.path.exists(entity_file):
            df_new.to_csv(entity_file, mode='a', header=False, index=False, encoding="utf-8-sig")
        else:
            df_new.to_csv(entity_file, index=False, encoding="utf-8-sig")
        
        logger.info(f"追加 {len(entities)} 个新实体")
    
    if relations:
        # 准备关系数据
        relation_data = []
        for relation in relations:
            relation_data.append({
                "relation_id": f"R_{relation.content_hash[:8]}",
                "source_entity": relation.entity1,
                "target_entity": relation.entity2,
                "relation_type": relation.relation_type,
                "confidence": relation.confidence,
                "attributes": json.dumps(relation.attributes, ensure_ascii=False),
                "source_file": relation.source_file,
                "content_hash": relation.content_hash,
                "create_time": dt.datetime.now().isoformat()
            })
        
        # 追加到CSV
        relation_file = os.path.join(OUT_DIR, "bearing_process_relations.csv")
        df_new = pd.DataFrame(relation_data)
        
        if os.path.exists(relation_file):
            df_new.to_csv(relation_file, mode='a', header=False, index=False, encoding="utf-8-sig")
        else:
            df_new.to_csv(relation_file, index=False, encoding="utf-8-sig")
        
        logger.info(f"追加 {len(relations)} 个新关系")

async def process_bearing_documents_batch(pdf_dir: str, max_files: int = None):
    """批量处理滑动轴承工艺文档"""
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # 获取PDF文件
    pdf_files = glob.glob(os.path.join(pdf_dir, "**/*.pdf"), recursive=True)
    if not pdf_files:
        logger.error("未找到PDF文件")
        return
    
    if max_files:
        pdf_files = pdf_files[:max_files]
    
    logger.info(f"找到 {len(pdf_files)} 个PDF文件")
    
    # 加载现有数据
    existing_entity_hashes, existing_relation_hashes = load_existing_data()
    
    extractor = BearingProcessExtractor()
    processed_count = 0
    skipped_count = 0
    
    async with httpx.AsyncClient(http2=True, timeout=None) as client:
        for pdf_file in pdf_files:
            file_name = os.path.basename(pdf_file)
            logger.info(f"处理文件 ({processed_count + skipped_count + 1}/{len(pdf_files)}): {file_name}")
            
            # 检查文件是否已处理
            file_hash = calculate_file_hash(pdf_file)
            if extractor.file_tracker.is_processed(pdf_file, file_hash):
                logger.info(f"跳过已处理文件: {file_name}")
                skipped_count += 1
                continue
            
            # 提取PDF文本
            full_text = extract_text_enhanced(pdf_file)
            if not full_text:
                logger.warning(f"无法提取文本: {file_name}")
                continue
            
            # 智能分块
            chunks = split_text_intelligently(full_text)
            logger.info(f"分割为 {len(chunks)} 个文本块")
            
            # 处理文本块
            file_entities = []
            file_relations = []
            
            # 并发处理文本块
            tasks = [call_llm_with_retry(chunk, client) for chunk in chunks]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理响应
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    logger.error(f"处理块 {i} 时出错: {response}")
                    continue
                
                if not response:
                    continue
                
                # 处理实体
                for entity_data in response.get("entities", []):
                    entity = ProcessEntity(
                        text=entity_data.get("text", ""),
                        entity_type=entity_data.get("type", ""),
                        confidence=entity_data.get("confidence", 0.5),
                        attributes=entity_data.get("attributes", {}),
                        start_pos=entity_data.get("start", 0),
                        end_pos=entity_data.get("end", 0),
                        source_file=file_name
                    )
                    file_entities.append(entity)
                
                # 处理关系
                for relation_data in response.get("relations", []):
                    relation = ProcessRelation(
                        entity1=relation_data.get("entity1", ""),
                        entity2=relation_data.get("entity2", ""),
                        relation_type=relation_data.get("type", ""),
                        confidence=relation_data.get("confidence", 0.5),
                        attributes=relation_data.get("attributes", {}),
                        source_file=file_name
                    )
                    file_relations.append(relation)
            
            # 提取数值参数
            numeric_entities = extractor.extract_numerical_parameters(full_text)
            for entity in numeric_entities:
                entity.source_file = file_name
            file_entities.extend(numeric_entities)
            
            # 提取工艺序列
            sequence_relations = extractor.extract_process_sequences(full_text)
            for relation in sequence_relations:
                relation.source_file = file_name
            file_relations.extend(sequence_relations)
            
            # 去重并过滤新数据
            unique_entities, unique_relations = merge_and_deduplicate_advanced(
                file_entities, file_relations, 
                existing_entity_hashes, existing_relation_hashes
            )
            
            # 追加到CSV
            if unique_entities or unique_relations:
                append_to_csv(unique_entities, unique_relations)
                
                # 更新已存在的哈希集合
                existing_entity_hashes.update(e.content_hash for e in unique_entities)
                existing_relation_hashes.update(r.content_hash for r in unique_relations)
                
                logger.info(f"文件 {file_name} 处理完成: {len(unique_entities)} 个新实体, {len(unique_relations)} 个新关系")
            else:
                logger.info(f"文件 {file_name} 无新数据")
            
            # 标记文件为已处理
            extractor.file_tracker.mark_processed(
                pdf_file, file_hash, len(unique_entities), len(unique_relations)
            )
            
            processed_count += 1
            
            # 添加延迟避免API限流
            await asyncio.sleep(1)
    
    logger.info(f"批量处理完成: 处理 {processed_count} 个文件, 跳过 {skipped_count} 个文件")
    
    # 生成处理报告
    generate_batch_report(processed_count, skipped_count)

def generate_batch_report(processed_count: int, skipped_count: int):
    """生成批量处理报告"""
    
    # 统计当前数据
    entity_file = os.path.join(OUT_DIR, "bearing_process_entities.csv")
    relation_file = os.path.join(OUT_DIR, "bearing_process_relations.csv")
    
    total_entities = 0
    total_relations = 0
    
    if os.path.exists(entity_file):
        df = pd.read_csv(entity_file)
        total_entities = len(df)
    
    if os.path.exists(relation_file):
        df = pd.read_csv(relation_file)
        total_relations = len(df)
    
    report = f"""
# 滑动轴承工艺知识图谱批量处理报告

## 处理统计
- 新处理文件: {processed_count}
- 跳过文件: {skipped_count}
- 总实体数: {total_entities}
- 总关系数: {total_relations}

## 数据质量
- 平均每文件实体数: {total_entities / max(processed_count, 1):.1f}
- 平均每文件关系数: {total_relations / max(processed_count, 1):.1f}

## 去重效果
- 启用了基于内容哈希的去重
- 启用了语义相似度去重
- 支持增量更新

处理时间: {dt.datetime.now().isoformat()}
"""
    
    with open(os.path.join(OUT_DIR, "batch_processing_report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    
    logger.info("批量处理报告已生成")

if __name__ == "__main__":
    # 批量处理PDF文档
    # 可以指定最大文件数进行测试
    asyncio.run(process_bearing_documents_batch("./pdf", max_files=None))