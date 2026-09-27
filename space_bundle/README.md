---
title: AI Music Rights Recognition
emoji: 🎵
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
short_description: Nhận diện bản nhạc và đánh giá rủi ro bản quyền khi sử dụng
---

# Hệ thống AI nhận diện nhạc & đánh giá rủi ro bản quyền

Tải lên một file audio hoặc video. Hệ thống nhận diện bản nhạc qua ba tầng nối tiếp:
dấu vân tay Chromaprint → embedding MERT-v1-95M + FAISS → Cover (chroma/OTI). Sau đó
Rule Engine đánh giá mức rủi ro khi sử dụng: LOW / CONDITIONAL / HIGH / UNKNOWN.

- **Kết quả là đánh giá rủi ro kỹ thuật, không phải tư vấn pháp lý.**
- **Kho tham chiếu có 24.375 bài của Free Music Archive (Creative Commons).** Bài ngoài
  kho này sẽ ra UNKNOWN.
- **File tải lên không được lưu.** Bài chưa nhận diện được thì hệ thống giữ lại dấu vân
  tay và đặc trưng âm thanh (không giữ file) trong sổ bài chưa nhận diện. Người dùng
  trang này đều tra được sổ, nhưng tên file người khác đã gửi thì bị ẩn.
- Space chạy trên CPU miễn phí, nên chậm: một file 45 giây mất khoảng nửa phút, bài dài
  thì lâu hơn. Space ngủ sau 48 giờ không có ai dùng; lần mở kế tiếp phải chờ vài phút để
  khởi động lại.

Mã nguồn và tài liệu đầy đủ nằm trong repo GitHub của dự án. Space này được GitHub
Actions tự triển khai lại mỗi lần có commit mới trên nhánh `main`; đừng sửa file trực
tiếp trên Space.
