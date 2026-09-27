# Design System — giao diện Music Rights AI

Tài liệu cho người sửa `frontend/`. Mọi token nằm ở đầu [`frontend/styles.css`](../frontend/styles.css), font nằm trong [`frontend/fonts/`](../frontend/fonts/), dự án không có bước build. File CSS là nguồn sự thật; tài liệu này giải thích **vì sao** mỗi giá trị như vậy và mỗi thành phần phải trông ra sao ở từng trạng thái.

**Phong cách: "Aurora"** (2026-09-27). Chọn qua skill ui-ux-pro-max: phong cách "Aurora UI" ghi rõ hợp với nền tảng âm nhạc và hỗ trợ nền sáng. Đi kèm là gradient tím → hồng → cam trên nền trắng ánh tím, tiêu đề đậm, cùng các chi tiết âm nhạc: đĩa than, equalizer, sóng âm. Bộ design system tự sinh của skill ra phong cách nền tối (Dark Mode OLED) và phong cách cho trẻ em (Claymorphism), đều trái brief "tươi sáng, không tối màu", nên không dùng. Bảng màu dưới đây tự dựng và đo tương phản từng cặp.

---

## 1. Ba lớp token

| Lớp | Tiền tố | Ví dụ | Ai được tham chiếu |
|---|---|---|---|
| 1. Nguyên thuỷ | `--gray-*`, `--lilac-*`, `--violet-*`, `--pink-*`, `--orange-*`, `--green-*`, `--amber-*`, `--red-*`, `--space-*`, `--font-size-*`, `--radius-*`… | `--violet-600: #7c3aed` | Chỉ lớp 2 (với **màu**). Thang khoảng cách / chữ / bo góc thì rule gọi thẳng được. |
| 2. Ngữ nghĩa | `--color-*`, `--gradient-*`, `--aurora-1..3`, `--shadow-raised` | `--color-accent: var(--violet-600)` | Rule và lớp 3 |
| 3. Thành phần | `--button-*`, `--input-*`, `--card-*`, `--tab-*`, `--pill-*`, `--table-*`, `--tooltip-*`, `--chart-*`, `--vinyl-*`, `--eq-*`, `--wave-*`… | `--button-primary-bg: var(--gradient-cta)` | Rule của đúng thành phần đó |

Quy tắc:
- **Không viết mã màu trong rule.** Cũng không gọi thẳng dải màu (`var(--violet-600)`) trong rule, mà phải đi qua `--color-*` / `--gradient-*`.
- **Chế độ tối chỉ ghi đè lớp 2**, trong `@media (prefers-color-scheme: dark)`. Brief yêu cầu nền sáng làm chủ đạo, nên chế độ sáng là mặc định; chế độ tối chỉ bật khi hệ điều hành của người dùng chọn tối.
- **Biến thể = đặt lại token cục bộ**, không viết lại cả rule. Ví dụ `.risk-HIGH { --risk-fill: …; --risk-text: …; }` dùng chung một rule `.risk-banner`.
- **Canvas đọc token**: sóng âm vẽ bằng canvas lấy màu qua `getComputedStyle` từ `--color-wave-from/to/empty`, không có mã màu trong JS.

Script kiểm nhanh (chạy từ gốc repo):

```bash
python - <<'EOF'
import re
css = open('frontend/styles.css', encoding='utf8').read()
js = open('frontend/app.js', encoding='utf8').read()
used = set(re.findall(r'var\((--[\w-]+)', css + js)) | set(re.findall(r"read\('(--[\w-]+)'\)", js))
defined = set(re.findall(r'(--[\w-]+)\s*:', css))
rules = css[css.index('/* ══ NỀN'):]
print('undefined:', used - defined - {'--i'})
print('unused:', defined - used)
print('colour literals in rules:', re.findall(r'#[0-9a-fA-F]{3,6}\b|rgba?\(', rules))
print('primitive colour refs in rules:', set(re.findall(r'var\(--(?:gray|lilac|violet|pink|orange|green|amber|red|ink)-\d+', rules)))
EOF
```

Cả bốn dòng phải in ra tập / danh sách rỗng. `--i` là số thứ tự do JS đặt trên từng thẻ kết quả để các thẻ hiện lần lượt, có giá trị dự phòng 0.

---

## 2. Màu

Tỉ lệ tương phản đo theo WCAG 2.x trên nền `surface` (và `bg` nếu khác). Chữ thường cần ≥ 4.5:1; chữ lớn (≥ 24px, hoặc ≥ 18.66px đậm) cần ≥ 3:1. Viền ô nhập, thanh, chấm và dải màu cần ≥ 3:1.

### Nền, chữ, viền

| Token | Sáng | Tối | Dùng cho | Tương phản (sáng / tối) |
|---|---|---|---|---|
| `--color-bg` | lilac-50 `#f7f5fc` | gray-900 `#14171c` | nền trang, nền ô nhập | — |
| `--color-surface` | gray-0 `#ffffff` | gray-800 `#1c2027` | thẻ, tab, hero | — |
| `--color-border` | lilac-200 `#e6e0f2` | gray-700 `#2c323b` | viền **trang trí** (thẻ, bảng, nút phụ) | không cần đạt |
| `--color-border-accent` | violet-200 `#ddd6fe` | violet-800 `#5b21b6` | viền nét đứt của vùng thả file | trang trí (1.39) — vùng thả còn chữ + nút |
| `--color-border-strong` | gray-500 `#878e98` | gray-600 `#646b75` | viền **nhận diện** ô nhập / select, vòng bước chưa tới | 3.31 (3.06 trên bg) / 3.04 |
| `--color-text` | gray-800 `#1c2027` | gray-100 `#e8eaed` | chữ chính | 16.3 (15.1 trên bg) / 13.6 |
| `--color-text-muted` | gray-600 `#646b75` | gray-400 `#9aa3ae` | nhãn, gợi ý | 5.38 (4.98 trên bg) / 6.40 |

### Màu nhấn và gradient thương hiệu

| Token | Sáng | Tối | Dùng cho | Tương phản |
|---|---|---|---|---|
| `--color-accent` | violet-600 `#7c3aed` | violet-400 `#a78bfa` | thanh biểu đồ, vòng focus, viền bước đang mở | 5.70 trên trắng, 5.27 trên bg / 6.00 |
| `--color-accent-strong` | violet-700 `#6d28d9` | violet-400 | **chữ** màu nhấn: link, pill, eyebrow, hover nút phụ | 7.10 trắng · 6.57 bg · 6.48 accent-soft / 6.00 · 6.60 bg · 5.29 accent-soft |
| `--color-accent-soft` | violet-50 `#f5f3ff` | violet-950 `#2b2350` | nền pill, hàng đang chọn, chip đang bật | — |
| `--color-on-accent` | trắng | ink-950 `#0b1220` | chữ trên nền accent **đặc** (link bỏ qua, nút phản hồi đang chọn) | 5.70 / 6.88 |
| `--gradient-cta` | violet-600 → pink-600 `#db2777` | (giữ nguyên) | nền nút chính, tab đang mở, icon tính năng, vòng số | chữ trắng (`--color-on-gradient`) ≥ **4.60** ở mọi điểm |
| `--gradient-cta-hover` | violet-700 → pink-700 | (giữ nguyên) | nút chính khi hover | chữ trắng ≥ 6.04 |
| `--gradient-text` | violet-700 → pink-700 → orange-700 `#c2410c` | violet-300 → pink-400 → orange-400 | chữ gradient ở tiêu đề Trang chủ | ≥ 5.18 / ≥ 6.17 |
| `--gradient-brand` | violet-600 → pink-600 → orange-500 `#f97316` | (giữ nguyên) | **chỉ trang trí**: logo, cột equalizer, nhãn đĩa than, vạch trên ô số liệu | — |
| `--color-wave-from/to` · `--color-wave-empty` | violet-600 / pink-600 · violet-200 | violet-400 / pink-400 · gray-700 | sóng âm canvas: phần đã tô · phần chưa tới | đã tô vs chưa tô 4.10 / 3.31 · 4.74 / 4.87 |

Vì sao **màu cam không bao giờ nằm dưới chữ**: chữ trắng trên `#f97316` chỉ đạt 2.8:1, trên `#ea580c` chỉ 3.56:1. Gradient của nút chính vì thế chỉ đi từ tím sang hồng. Màu cam chỉ xuất hiện ở gradient trang trí và ở chữ tiêu đề lớn, ở đó dùng `#c2410c` (5.18:1).

Vì sao sóng âm **cũng bỏ màu cam**: đoạn cam đã tô so với phần chưa tô chỉ đạt 2.02:1 ở chế độ sáng. Ở chế độ tối, bản đầu dùng violet-600 trên violet-800 chỉ đạt 1.58:1, gần như không thấy đã chạy tới đâu. Nay chế độ tối dùng các sắc 400 sáng hơn trên nền xám.

### Mức rủi ro (không đổi, tách hẳn khỏi màu thương hiệu)

| Token | Sáng | Tối | Dùng cho | Tương phản (sáng / tối) |
|---|---|---|---|---|
| `--color-{low,conditional,high,unknown}` | green-600 `#1a8f4c`, amber-600 `#b8770a`, red-600 `#cf2a2a`, gray-600 | các bước -400 | **tô**: thanh, chấm, dải trái banner, vòng quanh biểu tượng mức | ≥ 3.70 / ≥ 5.4 |
| `--color-{…}-strong` | green-700 `#157a45`, amber-700 `#935e07`, red-600, gray-600 | = bản thường | **chữ** mức rủi ro, "Có/Không", nền số bước đã xong | ≥ 4.88 / ≥ 5.4 |

Không dùng hồng hay tím để báo rủi ro. Màu đỏ của mức HIGH đứng cạnh hồng thương hiệu dễ bị đọc nhầm, nên mọi chỗ báo rủi ro luôn kèm chữ (LOW / CONDITIONAL / HIGH / UNKNOWN) hoặc chấm màu kèm chữ.

**Thêm màu mới**: thêm bước vào dải ở lớp 1, trỏ một token lớp 2 vào nó, đo tương phản ở **cả hai** chế độ rồi ghi số vào bảng này.

---

## 3. Chữ, thang, chuyển động

### Font: Be Vietnam Pro (tự lưu, SIL OFL 1.1)

- `frontend/fonts/`: 4 độ đậm (400, 500, 600, 800). Mỗi độ đậm tách làm 3 phần theo `unicode-range` (vietnamese / latin / latin-ext), tổng 185 KB, và trình duyệt chỉ tải phần chứa ký tự cần dùng. Giấy phép nằm ở `fonts/OFL.txt`.
- Vì sao không dùng cặp font "Music/Entertainment" mà skill gợi ý (Righteous + Poppins): cơ sở dữ liệu font của skill cho thấy cả hai **không có bộ ký tự tiếng Việt**. Dấu ặ, ễ, ở sẽ bị lấy từ font khác và lệch nét. Be Vietnam Pro được thiết kế riêng cho dấu tiếng Việt chồng nhiều tầng.
- Lưu trong repo thay vì tải từ Google Fonts: chạy offline được và không gửi IP người dùng cho bên thứ ba. `backend/main.py` đăng ký kiểu MIME `font/woff2`, vì bảng MIME của Windows không có đuôi này.

### Thang

| Thang | Giá trị |
|---|---|
| Khoảng cách `--space-*` (rem, lưới 4px) | `0-5`=2 · `1`=4 · `1-5`=6 · `2`=8 · `2-5`=10 · `3`=12 · `3-5`=14 · `4`=16 · `5`=20 · `6`=24 · `8`=32 · `10`=40 · `12`=48 px |
| Cỡ chữ `--font-size-*` (rem) | `xs` 12 (nhỏ nhất) · `sm` 13 · `md` 14 · `base` 15 (thân bài) · `lg` 17 · `xl` 20 · `2xl` 24 (tiêu đề màn, mức rủi ro) · `3xl` 36 (chỉ số đếm) · `display` 32–52 theo bề rộng màn hình (tiêu đề Trang chủ, `clamp()`) |
| Độ đậm | `regular` 400 · `medium` 500 · `semibold` 600 · `bold` **800** (tiêu đề đậm theo brief) |
| Bo góc `--radius-*` | `xs` 4 · `md` 8 (ô nhập) · `lg` 12 (logo, icon) · `xl` 16 (thẻ) · `2xl` 24 (hero, vùng thả, trạng thái rỗng) · `full` (nút, tab, chip) |
| Độ mờ | `--opacity-disabled` .45 · `--opacity-subdued` .55 |
| Lớp chồng | `--z-tooltip` 10 · `--z-popover` 20 (bảng trạng thái hệ thống, link bỏ qua) |

### Chuyển động (thay cho framer-motion; dự án không có bước build)

| Token | Giá trị | Dùng cho |
|---|---|---|
| `--duration-fast` / `-base` / `-slow` | 150 / 250 / 450 ms | đổi màu / nhấc thẻ, nút / màn hiện ra, thẻ kết quả hiện dần |
| `--ease-out` | `cubic-bezier(.22, 1, .36, 1)` | vào màn, hiện dần, hover |
| `--ease-spring` | `cubic-bezier(.34, 1.56, .64, 1)` | nảy nhẹ khi nhận file |
| `--vinyl-spin` | 2.4 s/vòng | đĩa than quay |
| `--press-scale` / `--hover-lift` | .97 / −2 px | nhấn thì co lại / hover thì nhấc lên |

- **Chuyển màn**: `.screen.active` chạy `screen-in` (trượt lên 8 px và rõ dần), đồng bộ với việc chuyển focus vào heading của màn. Không dùng View Transitions API: API đó đổi DOM bất đồng bộ, nên lệnh chuyển focus sẽ rơi vào heading còn đang ẩn.
- **Kết quả hiện dần**: mỗi `.reveal` trễ `--i × 70 ms`. Chỉ số lớn đếm chạy lên bằng `requestAnimationFrame` (ease-out, 900 ms); trình đọc màn hình đọc con số cuối nằm trong `.sr-only`, không đọc các số đang chạy.
- **Giảm chuyển động** (`prefers-reduced-motion: reduce`): mọi animation và transition còn 0.01 ms, một lần. Đĩa than đứng yên, cột equalizer đứng ở độ cao cố định khác nhau, số đếm hiện ngay con số cuối, nút không nhấc hay co khi bấm.

---

## 4. Thành phần và trạng thái

Thứ tự ưu tiên khi nhiều trạng thái cùng áp dụng: **tắt > nhấn > focus > hover > mặc định**. Mọi thứ bấm được dùng chung một vòng focus: `outline: 2px solid var(--color-focus-ring)` (tím, 5.27:1 trên nền), cách phần tử 2px, chỉ hiện khi đi bằng bàn phím (`:focus-visible`).

### Nút chính `.primary` (gradient) và nút phụ `.ghost`

Hai loại nút cùng bo tròn hoàn toàn và cùng `min-height: var(--control-height)` (44px, vùng chạm tối thiểu), nên nút và ô nhập đặt cạnh nhau luôn thẳng hàng. `.cta` là bản lớn 48px, dùng cho nút Trang chủ và nút PHÂN TÍCH.

| Thuộc tính | Mặc định | Hover | Nhấn | Focus | Tắt |
|---|---|---|---|---|---|
| Nền chính | `gradient-cta` | `gradient-cta-hover` + phát sáng tím, nhấc −2px, phóng 1.02 | co về .97, tắt phát sáng | như mặc định + vòng | mờ .45, không đổi khi hover |
| Chữ chính | trắng (600) | = | = | = | = |
| Nền phụ | `surface` | `surface` | `accent-soft`, co .97 | như mặc định + vòng | mờ .45 |
| Viền / chữ phụ | `border` / `text` | `accent-strong` / `accent-strong` | = hover | = | = mặc định |

`.btn-small` cao 36px, chỉ dùng cạnh tiêu đề. Nút phản hồi (`.feedback-buttons`) cao 44px; khi `aria-pressed="true"` thì tô `accent`.

### Ô nhập `.field input / select / textarea`

| Mặc định | Hover | Focus | Điện thoại (≤ 720px) |
|---|---|---|---|
| nền `bg`, viền `border-strong`, cao 44px | viền `text-muted` | vòng focus | chữ 16px, vì Safari iOS tự phóng to trang khi chữ trong ô < 16px |

### Thanh trên và điều hướng

| Thành phần | Mô tả |
|---|---|
| `.topbar` | Một hàng gồm logo, dải tab và trạng thái hệ thống. Ở ≤ 860px, tab xuống hàng 2; ở ≤ 560px, tab "Thống kê" bỏ đuôi "& Tra cứu". Tên ứng dụng là `h1` duy nhất của trang. |
| Tab `.tab` | Dải viên thuốc (4 tab: Trang chủ · Phân tích · Lịch sử · Thống kê & Tra cứu). Tab đang mở tô `--tab-indicator` (gradient CTA, chữ trắng ≥ 4.6:1) và mang `aria-current="page"`. |
| Thanh bước `.step` | Chỉ hiện ở 4 màn Phân tích. Bước chưa tới: vòng viền; đang ở: tô `accent`; xong: tô `low-strong`. Bước chưa tới được thì `disabled` nhưng không làm mờ. |
| Trạng thái hệ thống `.health` | `<details>` luôn ở góc phải; bảng chi tiết mở sang trái, Esc hoặc bấm ra ngoài thì đóng. |

### Âm nhạc

| Thành phần | Mô tả | Truy cập |
|---|---|---|
| Đĩa than `.vinyl` (`-sm`, `-still`) | Rãnh vẽ bằng `repeating-radial-gradient`, nhãn giữa là `gradient-brand`, quay 2.4 s/vòng. Bản `-still` (trạng thái rỗng của Lịch sử) đứng yên. | `aria-hidden` — thuần trang trí |
| Equalizer `.eq` | Các cột rộng cố định 8px (`--eq-bar`), nhún bằng `scaleY` với chu kỳ lệch nhau theo `nth-child`. Cột phải hẹp cố định: cột giãn theo bề ngang bị `scaleY` bóp thành hình bầu dục. | `aria-hidden` |
| Sóng âm của file `#file-wave` | **Biên độ thật** của file: giải mã ngay trong trình duyệt bằng `OfflineAudioContext`, không gửi đi đâu, rút 96 đỉnh. Chỉ giải mã file audio ≤ 15 MB (dữ liệu PCM sau giải mã lớn khoảng 10 lần MP3). Video hoặc file lớn hơn: không hiện. | `aria-hidden` |
| Thanh tiến trình sóng âm `#proc-progress` | Tô gradient tới **bước pipeline thật** do backend báo, không chạy theo đồng hồ. Không giải mã được file thì dùng sóng trang trí suy từ tên file (cố định cho mỗi file), không bao giờ giả là biên độ thật. | `role="progressbar"` + `aria-valuenow` = số bước, `aria-valuetext` = tên bước |

### Trang chủ, Kết quả, Lịch sử

| Thành phần | Mô tả |
|---|---|
| `.hero` | Thẻ lớn trên nền aurora: hai quầng sáng mờ (`--aurora-1..3`) trôi chậm 14–18 s phía sau nội dung. Tiêu đề dùng `grad-text`. `#hero-meta` hiện số bản ghi **thật** lấy từ `/health`. |
| `.feature-grid`, `.how-steps`, `.cta-band` | 4 thẻ tính năng (4 → 2 → 1 cột), mỗi thẻ có icon trên nền gradient CTA và nhấc lên + phát sáng khi hover. 3 bước "Cách hoạt động" có vòng số gradient. Dải CTA lặp lại ở cuối trang. |
| `.server-note` | Hộp ở đầu màn Tải lên, hiện khi `/health` báo máy chủ hạn chế: cùng dáng `.worst-case-banner` (dải trái `--risk-fill`). Sọc vàng `--color-conditional` = hạn chế (thiếu chỉ mục MERT, FFmpeg); thêm `.blocking` = sọc đỏ `--color-high` khi máy chủ chắc chắn từ chối mọi file (mất CSDL, Rule Engine). `role="status"`. Thay cho tablist "Tải file \| Dán link" đã bỏ ngày 2026-09-27. |
| Vùng thả `.dropzone` | Viền nét đứt `border-accent`. Hover: viền `accent` + nền `gradient-soft`. Kéo file vào: viền gradient liền, phóng 1.01, phát sáng, icon nhún. Nhận file: `.file-picked.received` nảy nhẹ (`--ease-spring`). |
| `.risk-banner` + `.risk-metric` | Biểu tượng mức nằm trong vòng màu `--risk-fill`. Bên phải là chỉ số lớn đếm chạy lên: độ tin cậy nhận diện **thật** của backend. Chưa định danh được thì nhãn ghi "Điểm cao nhất — dưới ngưỡng, chưa định danh được", để 7% không bị đọc thành "khớp 7%". |
| Thẻ ví dụ `.card-sample` (thể loại) | Ba dấu hiệu cho thấy đây không phải kết quả thật: viền nét đứt, huy hiệu "Ví dụ minh hoạ", thanh gạch chéo `.sample-bar`. Dữ liệu cố định, giống nhau cho mọi file, vì hệ thống **chưa** phân loại thể loại. Các tab thể loại là tablist ARIA; panel trượt vào khi đổi tab. |
| `.chip` | Chip bật/tắt dùng chung cho lọc Lịch sử (`aria-pressed`) và tab thể loại (`aria-selected`). Chip đang bật có nền `accent-soft`, viền `accent`, chữ `accent-strong`. |
| `.history-item` | Dải trái theo `--risk-fill`. Bên trong là một nút mở lại kết quả và một nút ✕ xoá (có `aria-label`), hover thì nhấc lên. Nếu job không còn trên máy chủ, lỗi hiện ngay trong mục đó. |
| `.empty-state` | Khung nét đứt, đĩa than đứng yên, câu hướng dẫn và nút CTA. |

### Các thành phần khác (giữ từ trước)

| Thành phần | Token | Ghi chú |
|---|---|---|
| Thẻ `.card`, ô số liệu `.stat` | `--card-*`: đệm 20, bo 16, cách nhau 16 | Ô số liệu có vạch `gradient-brand` 3px ở trên; con số vẫn dùng màu chữ, không dùng màu chuỗi |
| Pill `.pill` | nền `accent-soft`, chữ `accent-strong` (6.48:1) | |
| Banner `.risk-banner`, `.worst-case-banner` | dải trái 6px theo token cục bộ `--risk-fill`; chữ mức theo `--risk-text` | |
| Biểu đồ | một chuỗi = `--chart-fill` (tím); theo mức rủi ro thì dùng `--color-{mức}` và luôn kèm nhãn chữ | Quy tắc chi tiết: skill dataviz |
| Icon `.icon` | SVG nét 24×24, `stroke: currentColor`, `aria-hidden` | Không dùng emoji làm icon |
| `.sr-only`, `.skip-link` | ẩn khỏi mắt nhưng trình đọc màn hình vẫn đọc; link bỏ qua hiện ra khi nhận focus | Mỗi màn có một `h2` (`tabindex="-1"`) để nhận focus khi chuyển màn |

---

## 5. Chưa làm

- **Màn Kết quả vẫn dùng emoji cho mức rủi ro** (⚪🟢🟡🔴, theo CLAUDE.md §15). ⚪ hiện thành quả cầu tím trên Windows. Tab Thống kê và Lịch sử đã dùng chấm màu `.risk-dot`; màn Kết quả chưa đổi vì đổi là lệch khỏi đặc tả §15.
- **Chữ thân bài 15px** (skill khuyến nghị 16px trên điện thoại). Riêng ô nhập đã lên 16px trên điện thoại.
- **Phân loại thể loại** chỉ có giao diện, được ghi rõ là ví dụ. Muốn chạy thật cần một mô hình thể loại có thí nghiệm riêng, nằm ngoài pipeline lõi (CLAUDE.md §2.9). Phần **Dán link** đã bỏ hẳn (2026-09-27, chủ dự án chưa định phát triển).
- **Dashboard admin**: dùng tab Thống kê & Tra cứu (sổ bài chưa nhận diện) hiện có. Chưa có API thống kê toàn bộ job (tỉ lệ vi phạm), vì liệt kê job cần phân quyền (job_id là khoá truy cập).
- **Không có file token JSON / bước sinh CSS.** Dự án cố ý không có bundler.
