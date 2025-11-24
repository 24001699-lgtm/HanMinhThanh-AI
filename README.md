

# 🤖 Báo cáo Bài tập nhóm Môn Trí tuệ Nhân tạo

**📋 Thông tin:**

[Các thông tin này cũng cần được đưa vào báo cáo PDF và slide trình bày.]

* **📚 Môn học:** [MAT3508] - Nhập môn Trí tuệ Nhân tạo  
* **📅 Học kỳ:** Học kỳ 1, Năm học 2025-2026 
* **🏫 Trường:** VNU-HUS (Đại học Quốc gia Hà Nội - Trường Đại học Khoa học Tự nhiên)  
* **📝 Tiêu đề:** Applying PhoBERT Encoder for Sentiment Classification  
* **📅 Ngày nộp:** 30/11/2025  
* **📄 Báo cáo PDF:** 📄 [Liên kết tới báo cáo PDF trong kho lưu trữ này]  
* **🖥️ Slide thuyết trình:** 🖥️ [Liên kết tới slide thuyết trình trong kho lưu trữ này]  
* **📂 Kho lưu trữ:** 📁 Bao gồm mã nguồn, dữ liệu và tài liệu (hoặc dẫn link ngoài nếu cần)

**👥 Thành viên nhóm:**

| 👤 Họ và tên      | 🆔 Mã sinh viên     | 🐙 Tên GitHub        | 🛠️ Đóng góp  |
|------------------|--------------------|----------------------|----------------------|
| Hán Minh Thành   | 24001699           | 24001699-lgtm        | Thực hiện dự án      |

---

## 📑 Tổng quan cấu trúc báo cáo

Báo cáo này trình bày quá trình nghiên cứu và ứng dụng mô hình PhoBERT cho bài
toán phân loại cảm xúc tiếng Việt, sử dụng dữ liệu phản hồi của khách hàng trong lĩnh
vực dược phẩm. Nội dung báo cáo bao gồm các giai đoạn: thu thập dữ liệu (crawl ), tiền xử
lý, gán nhãn bằng Doccano, gán nhãn thực thể (NER), tinh chỉnh (fine-tune) mô hình
PhoBERT .Kết quả thực nghiệm cho thấy PhoBERT đạt độ chính xác cao, hoạt động ổn định và có
tiềm năng ứng dụng thực tế trong các hệ thống phân tích cảm xúc tiếng Việt.

### Chương 1: Giới thiệu
**📝 Tóm tắt dự án**
   - Tổng quan: Dự án tập trung vào việc xây dựng và tối ưu hóa mô hình học sâu để giải quyết bài toán Phân tích cảm xúc trong tiếng Việt.

 - Mục tiêu: Tận dụng kiến trúc Transformer và mô hình ngôn ngữ tiền huấn luyện (Pre-trained Language Model) PhoBERT để đạt hiệu suất cao 

 - Kết quả: Xây dựng thành công mô hình có khả năng hiểu ngữ cảnh 

**❓ Bài toán đặt ra**
   - Vấn đề: Tiếng Việt có đặc thù về từ ghép và ngữ pháp phức tạp. Các mô hình cũ thường gặp khó khăn trong việc ghi nhớ ngữ cảnh dài và xử lý sự đa nghĩa của từ.

   - Ý nghĩa: Việc giải quyết bài toán này giúp tự động hóa quy trình chăm sóc khách hàng, giảm thiểu sức người và nâng cao độ chính xác trong xử lý dữ liệu văn bản lớn.

### Chương 2: Phương pháp & Triển khai
**⚙️ Phương pháp**
   -Cơ sở lý thuyết:

Kiến trúc Transformer: Sử dụng cơ chế Self-Attention (Query, Key, Value) để mô hình có thể "nhìn" toàn bộ câu cùng lúc, đánh giá trọng số quan trọng của từng từ dựa trên ngữ cảnh thay vì xử lý tuần tự.

PhoBERT: Sử dụng mô hình BERT đã được huấn luyện trước trên dữ liệu tiếng Việt khổng lồ (20GB văn bản), tích hợp cơ chế Next Sentence Prediction (NSP) để hiểu mối quan hệ logic giữa các câu và Masked Language Modeling (MLM).

Thuật toán tối ưu: Sử dụng AdamW (Adam with Decoupled Weight Decay).

Tách biệt phần suy giảm trọng số (Weight Decay) khỏi bước cập nhật gradient thích ứng.

Giúp mô hình tổng quát hóa tốt hơn (giảm Overfitting) và hội tụ ổn định hơn so với Adam thường.

Xử lý dữ liệu: Sử dụng bộ từ điển (Vocabulary) và Tokenizer của VinAI (PhoBERT).

Áp dụng kỹ thuật tách từ (Word Segmentation) tự động để khớp với chỉ số (Index ID) trong từ điển.

**💻 Triển khai**

   -Kiến trúc mã nguồn:

   Preprocessing: Chuẩn hóa văn bản, gán nhãn từ loại, chuyển đổi text sang Input IDs bằng VinAI Tokenizer.

   Model: Load pre-trained vinai/phobert-base, thêm lớp Linear (Fully Connected) ở đầu ra để phục vụ bài toán phân loại cụ thể.

   Training Loop: Cài đặt vòng lặp huấn luyện với hàm Loss (CrossEntropy) và tối ưu hóa bằng AdamW (Learning rate warm-up).

### Chương 3: Kết quả & Phân tích
**📊 Kết quả & Thảo luận**

   Chỉ số đánh giá: Sử dụng các độ đo Accuracy, Precision, Recall và F1-Score.

   Phân tích:

      Hiệu quả của Self-Attention: Mô hình nhận diện chính xác các từ khóa quan trọng trong câu dài mà không bị mất thông tin.

      Tốc độ hội tụ: Nhờ AdamW, biểu đồ Loss giảm đều và ổn định, tránh được các điểm cực tiểu địa phương tốt hơn.

   
      
### Chương 4: Kết luận
**✅ Kết luận & Hướng phát triển**
   - Tổng kết: Dự án đã chứng minh sức mạnh của việc kết hợp kiến thức đặc thù ngôn ngữ (PhoBERT) với các kỹ thuật tối ưu hiện đại (AdamW, Self-Attention) để giải quyết bài toán NLP tiếng Việt.

   - Đề xuất cải tiến:

      Thử nghiệm với phiên bản phobert-large để tăng độ chính xác.
      
      Tăng cường dữ liệu (Data Augmentation) để cải thiện khả năng chịu lỗi của mô hình.
      
      Tinh chỉnh siêu tham số (Hyperparameter tuning) kỹ hơn cho AdamW (learning rate, weight decay).

### Tài liệu tham khảo 
📚 Tài liệu tham khảo

   Vaswani et al. (2017). "Attention Is All You Need". (Cơ sở về Transformer & Self-Attention).
   
   Nguyen & Nguyen (2020). "PhoBERT: Pre-trained language models for Vietnamese". (Mô hình VinAI).
   
   Loshchilov & Hutter (2019). "Decoupled Weight Decay Regularization". (Thuật toán AdamW).

📎 Phụ lục

   Biểu đồ huấn luyện (Training/Validation Loss).
   
   Đoạn code minh họa cách mapping từ vựng sang chỉ số dùng vinai/phobert-base.




### 📋 Yêu cầu

- **Định dạng:**  
   + 🖨️ Báo cáo phải được đánh máy, trình bày rõ ràng và xuất ra định dạng PDF (khuyến nghị dùng LaTeX).  
   + 🔁 Một bản báo cáo cần lưu trên kho GitHub của dự án, hai bản nộp trên Canvas (một cho giảng viên, một cho trợ giảng), và hai bản in (một cho giảng viên, một cho trợ giảng). Slide trình bày cũng thực hiện tương tự (không cần bản in slides).
- **Kho lưu trữ:** 📂 Bao gồm báo cáo PDF, slide, toàn bộ mã nguồn và tài liệu liên quan. Nếu vượt quá giới hạn dung lượng của GitHub, có thể tải lên Google Drive hoặc Dropbox và dẫn link trong tài liệu.
- **Làm việc nhóm:** 🤝 Cần ghi rõ đóng góp của từng thành viên trong nhóm.
- **Tài liệu hóa mã nguồn:**  
   + 🧾 Có bình luận giải thích rõ các thuật toán/phần logic phức tạp  
   + 🧪 Docstring cho hàm/phương thức mô tả tham số, giá trị trả về và mục đích  
   + 📘 File README cho từng module mã nguồn, hướng dẫn cài đặt và sử dụng  
   + 📝 Bình luận inline cho các đoạn mã không rõ ràng

### ✅ Danh sách kiểm tra trước khi nộp
- [X] ✅ Đánh dấu X vào ô để xác nhận hoàn thành  
- [X] ✍️ Điền đầy đủ các mục trong mẫu README này  
- [X] 📄 Hoàn thiện báo cáo PDF chi tiết theo cấu trúc trên  
- [X] 🎨 Tuân thủ định dạng và nội dung theo hướng dẫn giảng viên  
- [X] ➕ Thêm các mục riêng của dự án nếu cần  
- [X] 🔍 Kiểm tra lại ngữ pháp, diễn đạt và độ chính xác kỹ thuật  
- [X] ⬆️ Tải lên báo cáo PDF, slide trình bày và mã nguồn  
- [X] 🧩 Đảm bảo tất cả mã nguồn được tài liệu hóa đầy đủ với bình luận và docstring  
- [X] 🔗 Kiểm tra các liên kết và tài liệu tham khảo hoạt động đúng

### 🏆 Tiêu chí đánh giá Bài tập nhóm

Xem 📄 [Rubrics.md](Rubrics.md) để biết chi tiết về tiêu chí đánh giá bài tập nhóm, bao gồm điểm tối đa cho từng tiêu chí và mô tả các mức độ đánh giá (Xuất sắc, Tốt, Cần cải thiện).


