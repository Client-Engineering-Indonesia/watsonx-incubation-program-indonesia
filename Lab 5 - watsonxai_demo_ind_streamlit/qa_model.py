#Model Knowledge

question_answer_id = """
Pertanyaan: 
Apa itu Tingkat Komponen Dalam Negeri?

Jawaban:
Tingkat Komponen Dalam Negeri (TKDN) adalah salah satu indikator yang digunakan untuk mengukur tingkat ketergantungan suatu negara terhadap komponen dalam negeri. TKDN mencakup berbagai macam komponen, seperti bahan baku, tenaga kerja, teknologi, dan pengelolaan.
TKDN dapat dihitung dengan menggunakan rumus berikut:
TKDN = (Bahan Baku Dalam Negeri + Tenaga Kerja Dalam Negeri + Teknologi Dalam Negeri + Pengelolaan Dalam Negeri) / Total Nilai Produk Domestik Bruto (NPD)
Nilai TKDN yang lebih tinggi menunjukkan bahwa suatu negara lebih mengandalkan komponen dalam negeri untuk memproduksi barang dan jasa. Sebaliknya, nilai TKDN yang lebih rendah menunjukkan bahwa suatu negara lebih mengandalkan komponen luar negeri untuk memproduksi barang dan jasa.
TKDN dapat digunakan sebagai indikator untuk mengukur tingkat ketergantungan suatu negara terhadap komponen dalam negeri dan untuk menganalisis potensi risiko yang terkait dengan ketergantungan tersebut. 

Pertanyaan: 
Kenapa RUPS (Rapat Umum Pemegang Saham) Penting untuk diadakan?

Jawaban: 
RUPS (Rapat Umum Pemegang Saham) penting karena:
1. Pengambilan Keputusan Strategis: RUPS memungkinkan pemegang saham memutuskan arah dan kebijakan perusahaan.
2. Pertanggungjawaban dan Transparansi: RUPS menciptakan pertanggungjawaban dan transparansi dalam pengelolaan perusahaan.
3. Hak Partisipasi Pemegang Saham: Pemegang saham dapat berpartisipasi dan memberikan suara dalam keputusan perusahaan.
4. Kepentingan Pemegang Saham: Melindungi hak pemegang saham minoritas.
5. Kepatuhan Hukum: Mematuhi hukum dan peraturan yang mengharuskan perusahaan mengadakan RUPS.
6. Komunikasi dengan Pemegang Saham: Platform untuk komunikasi, menjelaskan kinerja, dan menjawab pertanyaan pemegang saham.
"""



def question_prompt(user_question, qa_sample):
    prompt = ""
    prompt += "<s>[INST] <<SYS>>\n"\
            "Selalu berikan jawaban yang dapat membantu, serta jawaban anda tidak boleh berbahaya, tidak etis, mengandung unsur rasis, dan sexist\n"\
            "Pastikan respon anda tidak bias dan juga memiliki makna positif.\n"\
            "Jawab pertanyaan dengan Bahasa Indonesia yang sesuai dengan Pedomanan Umum Ejaan Bahasa Indonesia serta pastikan jawaban anda padat dan jelas.\n"\
            "Pastikan untuk tidak menerjemahkan kata serapan ke dalam bahasa indonesia, biarkan dalam bahasa inggris.\n"\
            "Pastikan untuk tidak berhalusinasi, cukup jawab pertanyaan yang diajukan.\n"\
            "Berikut beberapa contoh yang dapat digunakan sebagai acuan:\n"\
            f"{qa_sample}\n"\
            "Apabila anda sama sekali tidak memiliki jawaban, jawab dengan mengatakan 'Saat ini saya membutuhkan informasi tambahan'.\n"\
            "<</SYS>>\n"\
            "Tolong jawab Pertanyaan dibawah ini dan pastikan jawaban dalam bahasa Indonesia:\n"\
            "[/INST]\n"\
            f"Pertanyaan: {user_question}\nJawaban: "
    return [prompt]


def answer_retrieval(model, prompt):
    answer_retrieval =  model.generate(prompt)
    result = answer_retrieval.generations[0][0].text
    return result

