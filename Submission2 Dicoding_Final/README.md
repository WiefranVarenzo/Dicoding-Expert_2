# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding

Jaya Jaya Institut merupakan salah satu institusi pendidikan perguruan yang telah berdiri sejak tahun 2000. Hingga saat ini ia telah mencetak banyak lulusan dengan reputasi yang sangat baik. Akan tetapi, terdapat banyak juga siswa yang tidak menyelesaikan pendidikannya alias dropout, yaitu persentase dropout mencapai 32,12%.

Jumlah dropout yang tinggi ini tentunya menjadi salah satu masalah yang besar untuk sebuah institusi pendidikan. Oleh karena itu, Jaya Jaya Institut ingin mendeteksi secepat mungkin siswa yang mungkin akan melakukan dropout sehingga dapat diberi bimbingan khusus.

Oleh sebab itu, dengan menggunakan model machine learning, diharapkan Jaya Jaya Institute bisa memprediksi mahasiswa mana yang memerlukan bimbingan khusus sebelum mereka terkena dropout.

### Permasalahan Bisnis

Jika Jaya Jaya Institute terus mengalami peristiwa dimana mahasiswanya banyak mengalami dropout, maka Jaya Jaya Institute bisa terkena teguran dari pihak dinas pendidikan ataupun mengalami pengurangan murid yang ingin masuk ke Institute ini. Jika terus berlanjut, bukan tidak mungkin bisa terkena sanksi ataupun permasalahan ekonomi, dikarenakan kurangnya mahasiswa yang mendaftar.

Kemudian, Nama dari Jaya-Jaya Institute ini akan semakin jelek di mata masyarakat, sehingga akan sulit mendapatkan kepercayaan masyarakat lagi. Dengan begitu, berbagai pihak yang bekerja pada institusi ini bisa mengalami pemberhentian kerja hingga berujung kepada penutupan dari institusi ini.

### Cakupan Proyek
Proyek ini bertujuan untuk menganalisis dan mengatasi masalah tingkat attrition yang tinggi di perusahaan JayaJaya Maju. Dengan mengidentifikasi faktor-faktor yang mempengaruhi keluarnya karyawan, proyek ini berusaha untuk mengembangkan strategi yang dapat mengurangi tingkat attrition dan meningkatkan retensi karyawan. Berikut adalah rincian cakupan proyek:

1. Analisis Data Dropout:
(1) Mengumpulkan dan membersihkan data mahasiswa dari Jaya Jaya Institute yang mencakup status menikah, SKS yang di ambil, SKS yang di setujui, nilai admisi, dan beberapa kolom lainnya yang berkaitan dengan data mahasiswa.
(2) Menganalisis data untuk menemukan pola dan tren yang mungkin berkontribusi terhadap tingginya tingkat Dropout.
2. Identifikasi Faktor Utama:
(1)Mengidentifikasi faktor-faktor utama yang mempengaruhi tingkat Dropout, seperti Nacionality, Curricular 2nd Semester Aprroved, Scholarship_holder, dan admission.
(2)Menggunakan teknik statistik dan machine learning untuk menentukan faktor yang paling signifikan.
3. Model Prediksi:
(1) Mengembangkan dua model prediksi untuk mengidentifikasi feature apa yang paling mempengaruhi prediksi, serta prediksi mahasiswa akan dropout atau tidak berdasarkan 15 feature yang paling utama.
(2) Model ini akan membantu institusi untuk memprediksi apakah mahasiswa akan dropout atau tidak, sehingga dapat melakukan bimbingan dengan lebih cepat.
4. Dashboard Visualisasi Data:
(1) Membangun dashboard interaktif yang menampilkan data Dropout Rate terkait tingkat dropout dan faktor-faktor yang mempengaruhinya.
(2) Dashboard ini akan membantu institusi dalam memantau tingkat dropout dan prediksi mahasiswa dropout yang dapat membantu untuk terjadinya pencegahan mahasiswa yang terkena dropout
5. Rekomendasi Strategi:
(1) Memberikan rekomendasi berbasis data untuk mengurangi tingkat dropout, seperti penyesuaian membuat komunitas mahasiswa (UKM) berkaitan dengan pelajaran sesuai jurusan, pembuatan program belajar intensif untuk mahasiswa yang memiliki nilai kurang memuaskan, pembuatan program ujian kembali (remedial), hingga melakukan seleksi berdasarkan nilai ujian masuk.
(2) Mengusulkan kebijakan dan program baru untuk meningkatkan mahasiswa yang lulus.

Output dari proyek ini meliputi:

1. Dua buah Model Prediksi: Model machine learning yang dapat memprediksi faktor-faktor yang memperbesar kemungkinan mahasiswa akan mengalami dropout, hingga memprediksi mahasiswa itu akan dropout atau tidak berdasarkan data yang diberikan.
2. Satu Dashboard Visualisasi Data: Dashboard interaktif yang menampilkan data terkait Dropout dan faktor-faktor yang mempengaruhi.
3. Satu Prototipe Machine Learning: Membuat UI dimana pihak Institusi bisa memasukkan data mahasiswa dan melihat prediksi apakah mahasiswa itu akan mengalami dropout atau tidak. Berikut adalah link Prototipenya (streamlit): https://dicoding-expert2-j8nmhls2haasnzaqured7j.streamlit.app/
4. Rekomendasi Strategi: Daftar rekomendasi strategi berbasis data untuk mengurangi tingkat dropout dan memperbesar persentasi mahasiswa yang lulus.
5. Laporan Analisis: Laporan komprehensif yang mencakup analisis data, temuan utama, dan rekomendasi tindakan.

Dengan proyek ini, diharapkan Jaya Jaya Institute dapat melakukan pencegahan peningkatan jumlah mahasiswa dropout, pemberian bimbingan khusus yang lebih tertarget, hingga meningkatkan persentasi mahasiswa lulus
### Persiapan
Sumber Data: 'https://github.com/dicodingacademy/dicoding_dataset/raw/main/students_performance/data.csv'

Setup environment:
```
python -m venv submission2
.\submission2\Scripts\activate
pip install joblib 
pip install scikit-learn == 1.2.2
pip freeze > requirements.txt
pip install -r requirements.txt

```

## Business Dashboard
link: https://public.tableau.com/views/JayaJayaInstitute/Dashboard1?:language=en-US&:sid=&:display_count=n&:origin=viz_share_link

Business Dashboard saya memiliki 7 bagan yang setiap bagannya memiliki representasi fakta pada data secara mandiri. Dimulai dari dari Dropout Rate yang merupakan bagan berisi nilai-nilai berkaitan dengan Ratio dari Dropout, dimana disini memiliki data jumlah total mahasiswa, Jumalah mahasiswa yang droput, jumlah mahasiswa yang lulus, serta persentasi dari tingkat dropout.

Kemudian, saya masuk ke faktor yang paling mempengaruhi dibandingkan dengan faktor lainnya, yaitu faktor Curricular 2nd Semester DropCount, dimana bisa dilihat bahwa mereka yang harga pembayarannya uptodate, namun unit kurikulum Semester 2 nya sedikit, maka akan mengalami dropout yang paling banyak. Sedangkan mereka yang harga pembayarannya tidak uptodate lebih sedikit daripada mereka yang harga pembayarannya up to date. Hal ini bisa dikarenakan mereka merasa harga pembayaran yang terbaru sangat mahal, sehingga mereka memiliki mengundurkan diri.

Pada bagan ketiga, bisa dilihat bahwa mereka yang merupakan orang portugis merupakan mahasiswa yang paling banyak, namun jumlah anak yang mengalami kelulusan dan yang dropout hampir sama (hampir 50%-50%), sehingga mereka yang berasal dari etnis portugis haruslah di perhatikan lebih.

Lalu, untuk mereka yang berkuliah pada siang hari memiliki jumlah mahasiswa dropout lebih besar dari pada mereka yang berkuliah pada malam hari. Hal ini berarti kita perlu lebih memfokuskan bimbingan kepada mereka yang berkuliah di siang hari.

Selanjutnya adalah bagan tingkat Dropout berdasarkan gender. Bisa dilihat disini bahwa mereka yang berkelamin wanita memiliki jumlah mahasiswa yang dropout lebih besar daripada mereka yang bergender pria. Oleh sebab itu, lebih memperhatikan mereka yang bergender wanita, terutama jika mereka memiliki gabungan faktor lain seperti merupakan orang portugis, serta harga pembayarannya up to date, dll, maka merekalah yang harus kita perhatikan lebih dan menerapkan bimbingan khusus kepada mereka.

Lalu yang keenam adalah pekerjaan ayah. Pekerjaan ayah pastinya akan berdampak pada ekonomi keluarga, lalu tingakt ekonomi keluarga pasti ada pengaruhnya dengan tingkat dropout seorang mahasiswa. Bisa dilihat jikalau mahasiswa yang memiliki ayah yang bekerja sebagai pekerja tidak berkeahlian (unskilled workers), maka merekalah yang paling banyak mengalami dropout. Hal ini bisa menjadi fokus utama untuk melakukan adanya bimbingan atau keringan lebih kepada mereka yang ayahnya memiliki pekerjaan kurang baik (seperti unskilled workers yang tentunya memiliki gaji kurang mencukupi)

Yang terakhir adalah data previous grade, dimana data inilah yang tentu saja memiliki tingkat pengaruh tidak sebesar yang lainnya, namun bisa dilihat bahwa mereka yang memiliki nilai tepat di rata-rata nilai total, apalagi mereka bukanlah mahasiswa international, maka paling banyak memiliki mahasiswa yang akan terkena dropout.

## Conclusion
Berdasarkan analisis data dropout di Jaya Jaya Institute, beberapa kesimpulan yang dapat diambil adalah sebagai berikut:

1. Curricular 2nd Semester DropCount: Mahasiswa yang memiliki SKS kurikulum semester 2 yang sedikit, terutama mereka yang pembayaran biayanya up-to-date, cenderung mengalami dropout lebih tinggi. Hal ini mungkin disebabkan oleh tekanan akademik atau finansial.

2. Nationality: Mahasiswa dari etnis Portugis memiliki tingkat dropout yang hampir setara dengan tingkat kelulusan, menunjukkan bahwa etnis ini memerlukan perhatian khusus untuk mengurangi dropout.

3. Time of Study: Mahasiswa yang berkuliah pada siang hari memiliki tingkat dropout lebih tinggi dibandingkan mereka yang berkuliah pada malam hari. Faktor ini menunjukkan perlunya fokus pada bimbingan di sesi kuliah siang.

4. Gender: Mahasiswa wanita memiliki tingkat dropout yang lebih tinggi dibandingkan mahasiswa pria, menunjukkan bahwa perlu ada perhatian khusus terhadap faktor-faktor yang mungkin lebih mempengaruhi wanita.

5. Father's Occupation: Mahasiswa dengan ayah yang bekerja sebagai pekerja tidak berkeahlian (unskilled workers) memiliki tingkat dropout yang lebih tinggi. Hal ini mungkin terkait dengan kondisi ekonomi keluarga yang kurang mendukung.

6. Previous Grade: Mahasiswa dengan nilai tepat di rata-rata nilai total memiliki kemungkinan dropout yang lebih tinggi. Hal ini menunjukkan bahwa prestasi akademik di masa lalu mempengaruhi tingkat dropout.


### Rekomendasi Action Items

1. Peningkatan Dukungan Akademik dan Bimbingan:

-) Program Bimbingan Intensif: Membuat program bimbingan akademik yang intensif untuk mahasiswa dengan SKS kurikulum semester 2 yang sedikit, terutama mereka yang pembayarannya up-to-date. Program ini bisa berupa sesi tutorial tambahan, bimbingan pribadi, atau kelompok belajar.

-) Remedial Programs: Menyediakan program remedial untuk membantu mahasiswa yang nilainya berada di rata-rata atau di bawah rata-rata, dengan fokus pada materi-materi yang sulit.

2. Peningkatan Dukungan Finansial:

-) Beasiswa dan Bantuan Keuangan: Mengembangkan program beasiswa atau bantuan keuangan khusus untuk mahasiswa dari latar belakang ekonomi yang kurang mampu, terutama mereka yang ayahnya bekerja sebagai unskilled workers.

-) Skema Pembayaran Fleksibel: Menawarkan skema pembayaran biaya kuliah yang lebih fleksibel untuk membantu mahasiswa mengatasi tekanan finansial.

3. Peningkatan Keterlibatan Mahasiswa:

-) Pengembangan Komunitas: Membentuk unit kegiatan mahasiswa (UKM) yang berkaitan dengan pelajaran sesuai jurusan untuk meningkatkan keterlibatan dan dukungan sosial antar mahasiswa.

-)Program Penghargaan: Mengadakan program penghargaan untuk mahasiswa yang menunjukkan prestasi akademik dan non-akademik sebagai bentuk apresiasi dan motivasi.

4. Penyesuaian Jadwal Kuliah:

-) Fleksibilitas Jadwal: Memberikan fleksibilitas lebih dalam pemilihan jadwal kuliah, terutama bagi mahasiswa yang kesulitan mengikuti perkuliahan siang hari.
-) Support Sessions: Mengadakan sesi dukungan khusus pada waktu siang hari untuk membantu mahasiswa yang berkuliah pada waktu tersebut.

5. Kebijakan dan Program Khusus untuk Gender:

-) Dukungan Khusus untuk Mahasiswi: Meningkatkan dukungan khusus bagi mahasiswi melalui program mentoring dan konseling untuk mengidentifikasi dan menangani masalah spesifik yang mungkin dihadapi mereka.

-) Pengarusutamaan Gender: Mengimplementasikan kebijakan pengarusutamaan gender dalam semua program dan kegiatan di institusi.

Dengan mengambil tindakan-tindakan ini, Jaya Jaya Institute dapat mengurangi tingkat dropout, meningkatkan tingkat kelulusan, dan memastikan mahasiswa mendapatkan dukungan yang mereka butuhkan untuk menyelesaikan pendidikan mereka dengan sukses.