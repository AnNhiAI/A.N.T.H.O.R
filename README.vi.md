# A.N.T.H.O.R

**AnNhi Thrifty High-Optimization Research**

Huấn luyện một Large Language Model từ đầu trên **một GPU NVIDIA T4 (16GB VRAM) duy nhất**.

## 🎯 Sứ mệnh

Dân chủ hóa việc huấn luyện LLM bằng cách chứng minh rằng một mô hình ngôn ngữ có năng lực hoàn toàn có thể được huấn luyện từ đầu trên phần cứng phổ thông. Dự án này khám phá giới hạn của **tối ưu cực đoan** — kết hợp các kỹ thuật tối ưu tiên tiến nhất, cải tiến kiến trúc, và chiến lược huấn luyện để tối đa hóa chất lượng mô hình dưới những ràng buộc khắc nghiệt về tài nguyên.

## 🧠 Tại sao là A.N.T.H.O.R?

Hầu hết các LLM đều cần hàng trăm GPU và hàng triệu đô la để huấn luyện. A.N.T.H.O.R thách thức mô hình này bằng câu hỏi: **Chúng ta có thể đạt được điều gì chỉ với một GPU T4?**

Đây vừa là một sân chơi nghiên cứu, vừa là một thử nghiệm thực tế về:

- **Tối ưu bộ nhớ** — đưa mô hình lớn vào 16GB VRAM
- **Tối ưu tính toán** — tối đa hóa lượng học được trên mỗi FLOP
- **Tối ưu dữ liệu** — tận dụng tối đa lượng dữ liệu tiền huấn luyện có hạn
- **Khám phá kiến trúc** — thử nghiệm các thiết kế mới ưu tiên hiệu suất tham số

## 🔧 Công nghệ sử dụng

| Thành phần | Cách tiếp cận |
|------------|---------------|
| **Kiến trúc** | Decoder-only Transformer với các cải tiến hiện đại (RoPE, SwiGLU, RMSNorm, Grouped-Query Attention) |
| **Độ chính xác** | Mixed precision (FP16/BF16), gradient checkpointing, có thể lượng tử hóa (FP8/INT8) |
| **Tối ưu hóa** | Bộ tối ưu tiết kiệm bộ nhớ (AdamW với bitsandbytes 8-bit, GaLore, huấn luyện kiểu LoRA) |
| **Song song hóa** | Đơn GPU với micro-batch accumulation, activation checkpointing |
| **Xử lý dữ liệu** | Tokenizer tùy chỉnh, dataloading hiệu quả với dataset ánh xạ bộ nhớ |
| **Giám sát** | Weights & Biases / TensorBoard để theo dõi quá trình huấn luyện |

## 🚀 Tính năng & Kỹ thuật

### Tối ưu bộ nhớ
- [ ] Activation checkpointing (gradient checkpointing)
- [ ] CPU offloading cho trạng thái optimizer
- [ ] Lượng tử hóa 4-bit / 8-bit (kiểu QLoRA)
- [ ] Huấn luyện tinh chỉnh tham số hiệu quả (LoRA, DoRA)
- [ ] GaLore (Gradient Low-Rank Projection)

### Hiệu suất huấn luyện
- [ ] Tích lũy vi mô-batch với gradient scaling
- [ ] Điều chỉnh kích thước batch động
- [ ] Lập lịch tốc độ học (cosine decay với warmup)
- [ ] Gradient clipping và chuẩn hóa
- [ ] Stochastic depth / layer dropout

### Dữ liệu & Tokenization
- [ ] Huấn luyện tokenizer BPE / Unigram tùy chỉnh
- [ ] Streaming dataset hiệu quả
- [ ] Khử trùng lặp và lọc chất lượng dữ liệu
- [ ] Học theo chương trình (sắp xếp dữ liệu từ dễ đến khó)

### Kiến trúc mô hình
- [ ] Rotary Position Embeddings (RoPE)
- [ ] Kích hoạt SwiGLU trong các tầng FFN
- [ ] Root Mean Square Normalization (RMSNorm)
- [ ] Grouped-Query Attention (GQA)
- [ ] Tùy chọn: Mixture of Experts (MoE)

## 📊 Tình trạng dự án

🚧 **Giai đoạn đầu** — Dự án đang trong giai đoạn nghiên cứu và tạo mẫu ban đầu.

- [ ] Nghiên cứu tài liệu & chọn lọc kỹ thuật
- [ ] Thiết kế kiến trúc & phân tích scaling laws
- [ ] Huấn luyện tokenizer
- [ ] Xây dựng pipeline dữ liệu
- [ ] Chạy huấn luyện quy mô nhỏ (chứng minh khái niệm)
- [ ] Chạy huấn luyện đầy đủ
- [ ] Đánh giá & so sánh
- [ ] Phát hành trọng số mô hình

## 🔬 Nhật ký nghiên cứu

Các phát hiện và thử nghiệm quan trọng sẽ được ghi lại trong quá trình phát triển.

## 📓 Notebooks (Google Colab)

Toàn bộ dự án được cấu trúc dưới dạng **Jupyter Notebooks** (`.ipynb`) được thiết kế để chạy trên **Google Colab** (gói T4 GPU miễn phí). Mỗi notebook tự chứa đầy đủ ô cài đặt, thiết lập, và thực thi — chỉ cần mở trong Colab và chạy.

```
ANTHOR/
├── README.md
├── README.vi.md
├── 01_tokenizer.ipynb       # Huấn luyện tokenizer BPE/Unigram
├── 02_data_pipeline.ipynb   # Xử lý dữ liệu & streaming
├── 03_model.ipynb           # Triển khai kiến trúc mô hình
├── 04_train.ipynb           # Vòng lặp huấn luyện & tối ưu
├── 05_eval.ipynb            # Đánh giá & benchmark
├── utils.ipynb              # Hàm tiện ích dùng chung
└── configs/                 # File cấu hình huấn luyện & mô hình
```

## 🖥️ Yêu cầu phần cứng

- **Tối thiểu:** NVIDIA T4 (16GB VRAM)
- **Mục tiêu:** Thiết lập GPU đơn
- **RAM:** Khuyến nghị 32GB+ bộ nhớ hệ thống
- **Lưu trữ:** 200GB+ dung lượng trống cho dataset & checkpoint

## 📚 Tài liệu tham khảo

- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [QLoRA: Efficient Finetuning of Quantized Language Models](https://arxiv.org/abs/2305.14314)
- [GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection](https://arxiv.org/abs/2403.03507)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)

## 📄 Giấy phép

Dự án này được cấp phép theo **GNU General Public License v3.0** — xem tệp [LICENSE](LICENSE) để biết chi tiết.

---

*Xây dựng với ❤️ bởi AnNhi — vì những điều lớn lao bắt đầu từ tài nguyên hạn chế.*
