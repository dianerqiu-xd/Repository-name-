import math
import random
import re
from collections import Counter
from typing import Dict, List, Tuple

import nltk
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


st.set_page_config(
    page_title="Week 7 语言模型训练与对比分析平台",
    page_icon="📘",
    layout="wide",
)

st.title("📘 Week 7 语言模型训练与对比分析平台")
st.caption("n-gram + RNN + 预训练模型 + 困惑度(PPL) 一体化实验")


def simple_tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\w\s]", text.lower())


def build_ngram_model(tokens: List[str], n: int) -> Tuple[Counter, Counter, List[Tuple[str, ...]], int]:
    padded = ["<s>"] * (n - 1) + tokens + ["</s>"]
    ngrams = [tuple(padded[i : i + n]) for i in range(len(padded) - n + 1)]
    ngram_counts = Counter(ngrams)
    context_counts = Counter(tuple(ng[:-1]) for ng in ngrams)
    vocab = set(tokens + ["</s>"])
    return ngram_counts, context_counts, ngrams, len(vocab)


def sentence_probability_details(
    sentence: str,
    n: int,
    ngram_counts: Counter,
    context_counts: Counter,
    vocab_size: int,
) -> Tuple[List[Dict[str, float]], float, float, float, float, bool]:
    tokens = simple_tokenize(sentence)
    if not tokens:
        return [], 0.0, 0.0, float("-inf"), float("-inf"), False

    padded = ["<s>"] * (n - 1) + tokens + ["</s>"]
    sent_ngrams = [tuple(padded[i : i + n]) for i in range(len(padded) - n + 1)]

    details: List[Dict[str, float]] = []
    log_unsmoothed = 0.0
    log_smoothed = 0.0
    has_zero = False

    for ng in sent_ngrams:
        context = ng[:-1]
        ng_count = ngram_counts.get(ng, 0)
        ctx_count = context_counts.get(context, 0)

        unsmoothed = (ng_count / ctx_count) if ctx_count > 0 else 0.0
        smoothed = (ng_count + 1.0) / (ctx_count + vocab_size)

        if unsmoothed <= 0.0:
            has_zero = True
        else:
            log_unsmoothed += math.log(unsmoothed)
        log_smoothed += math.log(smoothed)

        details.append(
            {
                "ngram": " ".join(ng),
                "count(ngram)": float(ng_count),
                "count(context)": float(ctx_count),
                "P_unsmoothed": unsmoothed,
                "P_add_one": smoothed,
            }
        )

    joint_unsmoothed = 0.0 if has_zero else math.exp(log_unsmoothed)
    joint_smoothed = math.exp(log_smoothed)
    log_unsmoothed_final = float("-inf") if has_zero else log_unsmoothed
    return details, joint_unsmoothed, joint_smoothed, log_unsmoothed_final, log_smoothed, has_zero


def pretty_prob(value: float) -> str:
    if value == 0.0:
        return "0"
    if math.isinf(value):
        return "∞"
    return f"{value:.3e}"


@st.cache_data(show_spinner=False)
def get_reuters_sample(max_words: int = 1200) -> str:
    try:
        nltk.data.find("corpora/reuters")
    except LookupError:
        nltk.download("reuters", quiet=True)

    try:
        from nltk.corpus import reuters

        words = list(reuters.words())[:max_words]
        text = " ".join(words)
        return text
    except Exception:
        return (
            "Natural language processing enables computers to understand text. "
            "Language models estimate how likely a sequence of words is. "
            "Smoothing techniques help when unseen n-grams appear."
        )


class CharRNNLM(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, model_type: str = "LSTM") -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.model_type = model_type
        if model_type == "RNN":
            self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        else:
            self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor, hidden=None):
        emb = self.embedding(x)
        out, hidden = self.rnn(emb, hidden)
        logits = self.fc(out)
        return logits, hidden


def build_char_dataset(text: str, seq_len: int):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    encoded = [stoi[c] for c in text]

    xs, ys = [], []
    for i in range(len(encoded) - seq_len):
        xs.append(encoded[i : i + seq_len])
        ys.append(encoded[i + 1 : i + seq_len + 1])
    return xs, ys, stoi, itos


def train_char_rnn(
    text: str,
    model_type: str,
    hidden_size: int,
    epochs: int,
    lr: float,
    seq_len: int,
    batch_size: int,
    progress_callback=None,
):
    xs, ys, stoi, itos = build_char_dataset(text, seq_len)
    if not xs:
        raise ValueError("语料太短，无法构建训练样本。请增加文本长度或减小 seq_len。")

    device = torch.device("cpu")
    x_tensor = torch.tensor(xs, dtype=torch.long, device=device)
    y_tensor = torch.tensor(ys, dtype=torch.long, device=device)
    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = CharRNNLM(vocab_size=len(stoi), hidden_size=hidden_size, model_type=model_type).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    loss_history: List[float] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits, _ = model(batch_x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), batch_y.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(loader))
        loss_history.append(avg_loss)
        if progress_callback is not None:
            progress_callback(epoch, avg_loss, loss_history)

    return model, stoi, itos, loss_history


def generate_from_char_model(
    model: CharRNNLM,
    stoi: Dict[str, int],
    itos: Dict[int, str],
    seed: str,
    max_new_chars: int = 50,
    temperature: float = 1.0,
) -> str:
    model.eval()
    device = torch.device("cpu")
    safe_temp = max(temperature, 1e-4)

    valid_seed = "".join(ch for ch in seed if ch in stoi)
    if not valid_seed:
        valid_seed = random.choice(list(stoi.keys()))

    generated = valid_seed
    hidden = None

    with torch.no_grad():
        for ch in valid_seed[:-1]:
            x = torch.tensor([[stoi[ch]]], dtype=torch.long, device=device)
            _, hidden = model(x, hidden)

        current = torch.tensor([[stoi[valid_seed[-1]]]], dtype=torch.long, device=device)
        for _ in range(max_new_chars):
            logits, hidden = model(current, hidden)
            next_logits = logits[:, -1, :] / safe_temp
            probs = torch.softmax(next_logits, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())
            next_char = itos[next_id]
            generated += next_char
            current = torch.tensor([[next_id]], dtype=torch.long, device=device)

    return generated


@st.cache_resource(show_spinner=False)
def load_bert_fill_mask_pipeline():
    return pipeline("fill-mask", model="bert-base-uncased")


@st.cache_resource(show_spinner=False)
def load_gpt2_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()
    return tokenizer, model


def gpt2_generate(prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> Tuple[str, str]:
    tokenizer, model = load_gpt2_model_and_tokenizer()
    device = torch.device("cpu")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=max(temperature, 1e-4),
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    continuation = text[len(prompt) :] if text.startswith(prompt) else text
    return text, continuation


def compute_gpt2_ppl(sentences: List[str]) -> List[Dict[str, float]]:
    tokenizer, model = load_gpt2_model_and_tokenizer()
    device = torch.device("cpu")
    rows = []

    for sentence in sentences:
        encoded = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        if input_ids.size(1) < 2:
            rows.append({"sentence": sentence, "cross_entropy_loss": float("nan"), "ppl": float("nan")})
            continue

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = float(outputs.loss.item())
        ppl = math.exp(loss) if loss < 80 else float("inf")
        rows.append({"sentence": sentence, "cross_entropy_loss": loss, "ppl": ppl})

    return rows


tab1, tab2, tab3, tab4 = st.tabs(
    [
        "模块1：n-gram & Smoothing",
        "模块2：Train your own RNN-LM",
        "模块3：Masked LM vs Causal LM",
        "模块4：Perplexity 困惑度",
    ]
)

with tab1:
    st.subheader("模块 1：n 元语言模型与加一平滑")
    source = st.radio(
        "语料来源",
        ["内置示例语料", "NLTK Reuters 采样语料", "手动输入语料"],
        horizontal=True,
    )

    builtin_text = (
        "I love natural language processing . "
        "Language models are useful for text prediction . "
        "I love building language models with Python ."
    )
    if source == "内置示例语料":
        corpus_text = st.text_area("语料内容", value=builtin_text, height=150, key="tab1_builtin")
    elif source == "NLTK Reuters 采样语料":
        corpus_text = st.text_area("语料内容（可编辑）", value=get_reuters_sample(), height=200, key="tab1_reuters")
        st.caption("提示：首次使用 Reuters 可能会自动下载语料。")
    else:
        corpus_text = st.text_area("语料内容（手动输入）", value="", height=200, key="tab1_custom")

    n = st.select_slider("选择 n-gram 阶数", options=[2, 3, 4], value=3)

    tokens = simple_tokenize(corpus_text)
    st.write(f"语料 Token 数：`{len(tokens)}`")
    if len(tokens) < n:
        st.warning(f"当前语料长度不足以构建 {n}-gram，请增加语料。")
    else:
        ngram_counts, context_counts, all_ngrams, vocab_size = build_ngram_model(tokens, n)
        st.write(f"词表大小 V：`{vocab_size}`，去重 n-gram 数：`{len(ngram_counts)}`")

        top_k = st.slider("展示高频 n-gram 数量", min_value=5, max_value=20, value=10, key="tab1_topk")
        top_rows = [
            {"n-gram": " ".join(ng), "count": c}
            for ng, c in ngram_counts.most_common(top_k)
        ]
        st.dataframe(top_rows, use_container_width=True)

        sentence = st.text_input(
            "输入句子，计算联合概率",
            value="I love language models .",
            key="tab1_sentence",
        )
        use_smoothing = st.checkbox("启用加一平滑（Add-one / Laplace）", value=False)

        if st.button("计算句子概率", key="tab1_compute"):
            details, p_unsm, p_sm, log_unsm, log_sm, has_zero = sentence_probability_details(
                sentence=sentence,
                n=n,
                ngram_counts=ngram_counts,
                context_counts=context_counts,
                vocab_size=vocab_size,
            )

            if not details:
                st.error("句子为空，无法计算。")
            else:
                selected_p = p_sm if use_smoothing else p_unsm
                selected_log = log_sm if use_smoothing else log_unsm
                st.metric("当前设置下的联合概率", pretty_prob(selected_p))
                st.caption(f"log 概率：{selected_log:.4f}" if not math.isinf(selected_log) else "log 概率：-∞")

                if has_zero and not use_smoothing:
                    st.warning("检测到未出现的 n-gram：未平滑模型联合概率归零。")

                st.write(
                    f"对比：未平滑 = `{pretty_prob(p_unsm)}`，加一平滑 = `{pretty_prob(p_sm)}`"
                )
                st.dataframe(details, use_container_width=True)

with tab2:
    st.subheader("模块 2：从零训练字符级 RNN 语言模型")
    default_train_text = "hello world hello world hello world hello world "
    train_text = st.text_area("输入训练语料（英文）", value=default_train_text, height=180)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        model_type = st.selectbox("RNN 类型", options=["LSTM", "RNN"], index=0)
        hidden_size = st.slider("Hidden Size", min_value=16, max_value=128, value=64, step=16)
    with col_b:
        epochs = st.slider("Epochs", min_value=10, max_value=200, value=60, step=10)
        seq_len = st.slider("Sequence Length", min_value=5, max_value=40, value=20, step=1)
    with col_c:
        lr = st.slider("Learning Rate", min_value=0.0005, max_value=0.05, value=0.01, step=0.0005)
        batch_size = st.select_slider("Batch Size", options=[8, 16, 32, 64], value=32)

    if st.button("开始训练", key="tab2_train"):
        if len(train_text) < seq_len + 1:
            st.error("训练文本太短，请增加文本或减小 Sequence Length。")
        else:
            progress_bar = st.progress(0)
            status = st.empty()
            chart_placeholder = st.empty()

            def on_progress(epoch: int, avg_loss: float, history: List[float]):
                progress_bar.progress(epoch / epochs)
                status.write(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f}")
                chart_placeholder.line_chart({"Loss": history})

            with st.spinner("训练中，请稍候..."):
                model, stoi, itos, loss_history = train_char_rnn(
                    text=train_text,
                    model_type=model_type,
                    hidden_size=hidden_size,
                    epochs=epochs,
                    lr=lr,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    progress_callback=on_progress,
                )

            st.session_state["rnn_model"] = model
            st.session_state["rnn_stoi"] = stoi
            st.session_state["rnn_itos"] = itos
            st.session_state["rnn_loss_history"] = loss_history
            st.success("训练完成。你现在可以输入 Seed 生成文本。")

    if "rnn_model" in st.session_state:
        st.markdown("#### 文本生成")
        seed = st.text_input("起始字符（Seed）", value="hello ")
        gen_len = st.slider("生成长度（新增字符数）", min_value=20, max_value=200, value=50, step=10)
        temp = st.slider("采样温度 Temperature", min_value=0.2, max_value=1.5, value=1.0, step=0.1)
        if st.button("生成文本", key="tab2_generate"):
            generated = generate_from_char_model(
                model=st.session_state["rnn_model"],
                stoi=st.session_state["rnn_stoi"],
                itos=st.session_state["rnn_itos"],
                seed=seed,
                max_new_chars=gen_len,
                temperature=temp,
            )
            st.code(generated)

with tab3:
    st.subheader("模块 3：预训练架构对比（BERT vs GPT-2）")
    st.caption("BERT：Masked LM（双向上下文）；GPT-2：Causal LM（左到右自回归）")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### BERT 填空（Masked LM）")
        bert_input = st.text_input(
            "输入含 [MASK] 的句子",
            value="The man went to the [MASK] to buy some milk.",
            key="tab3_bert_input",
        )
        if st.button("运行 BERT Top-5 预测", key="tab3_bert_btn"):
            if "[MASK]" not in bert_input:
                st.error("请输入包含 [MASK] 的句子。")
            else:
                try:
                    with st.spinner("加载/推理 bert-base-uncased ..."):
                        fill_mask = load_bert_fill_mask_pipeline()
                        preds = fill_mask(bert_input, top_k=5)
                    rows = [
                        {
                            "rank": i + 1,
                            "token": item["token_str"].strip(),
                            "probability": float(item["score"]),
                        }
                        for i, item in enumerate(preds)
                    ]
                    st.dataframe(rows, use_container_width=True)
                except Exception as exc:
                    st.error(f"BERT 推理失败：{exc}")

    with col_right:
        st.markdown("#### GPT-2 续写（Causal LM）")
        gpt_prompt = st.text_area(
            "输入前缀 Prompt",
            value="Artificial intelligence will change the future because",
            height=120,
            key="tab3_gpt_prompt",
        )
        max_new_tokens = st.slider("生成 token 数", min_value=10, max_value=60, value=20, step=5)
        gpt_temp = st.slider("Temperature", min_value=0.3, max_value=1.5, value=0.9, step=0.1)
        gpt_top_p = st.slider("Top-p", min_value=0.5, max_value=1.0, value=0.95, step=0.05)

        if st.button("运行 GPT-2 续写", key="tab3_gpt_btn"):
            try:
                with st.spinner("加载/推理 gpt2 ..."):
                    full_text, continuation = gpt2_generate(
                        prompt=gpt_prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=gpt_temp,
                        top_p=gpt_top_p,
                    )
                st.markdown("**续写结果（仅新增部分）**")
                st.write(continuation.strip() if continuation.strip() else "(模型未新增可见文本)")
                with st.expander("查看完整输出（含原始 Prompt）"):
                    st.write(full_text)
            except Exception as exc:
                st.error(f"GPT-2 推理失败：{exc}")

with tab4:
    st.subheader("模块 4：基于 GPT-2 的困惑度（Perplexity）计算")
    st.caption("按行输入多个句子，每行一个。系统计算 Cross-Entropy Loss 与 PPL = exp(Loss)。")

    default_eval = (
        "The weather is nice and I will go for a walk in the park.\n"
        "milk quickly blue tomorrow jumped because keyboard sleeps."
    )
    eval_text = st.text_area("测试句子列表（每行一条）", value=default_eval, height=180, key="tab4_input")

    if st.button("计算 PPL", key="tab4_calc"):
        sentences = [line.strip() for line in eval_text.splitlines() if line.strip()]
        if not sentences:
            st.error("请至少输入一条句子。")
        else:
            try:
                with st.spinner("计算中（首次会加载 gpt2）..."):
                    rows = compute_gpt2_ppl(sentences)
                st.dataframe(rows, use_container_width=True)
            except Exception as exc:
                st.error(f"PPL 计算失败：{exc}")

st.markdown("---")
st.info(
    "建议：先在模块1/2完成本地训练观察，再在模块3/4对比预训练模型行为。"
    "首次运行 transformers 模型需要联网下载。"
)
