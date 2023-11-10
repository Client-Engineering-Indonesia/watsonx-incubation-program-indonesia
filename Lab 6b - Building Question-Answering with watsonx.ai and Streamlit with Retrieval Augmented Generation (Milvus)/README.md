# Building  Apps with watsonx.ai and Streamlit
So I'm guessing you've been hearing a bit about watsonx. Well...now you can build your very own app with it 🙌 (I know...crazy right?!). In this tutorial you'll learn how to build your own LLM powered Streamlit with the Watson Machine Learning library.  

# Startup 🚀
1. Open your terminal or console window
2. cd into this lab's base directory
3. Copy your .env file into this lab's base folder
4. Make sure to have docker version > 19.0 installation in local
5. In Lab 6b, go to milvus directory. Then "sudo docker-compose -f docker-compose.yml up"
6. Check milvus in http://localhost:8000/ if it is successfully build
7. Go back to Lab 6b. Please to download the file for wikihow.csv on "[[https://ibm.box.com/s/8nvanf974t35d89cmibk75e3gc6d1pbo](https://ibm.box.com/s/8nvanf974t35d89cmibk75e3gc6d1pbo)](https://ibm.ent.box.com/folder/234332828731?s=8nvanf974t35d89cmibk75e3gc6d1pbo)" since the file is quite huge (1.2 GB)
8. Please read all the instructtion in milvus-demo.ipynb and run it in order to store and load your data in milvus
9. Check milvus attu to see is your data successfully upload or not
10. Run the app by running the command `streamlit run app.py`
11. You can ask the question with the question below

# Contoh Pertanyaan:
- Bagaimana caranya agar saya dapat memanjat pohon dengan selamat?
- Bagaimana cara menjelaskan kualifikasi dan pengalaman kita dalam pembuatan surat perkenalan (introductory letter)?
- Bagaimana cara menulis surat pengantar yang efektif, terutama dalam konteks surat pengantar bisnis, dan apa yang harus dihindari dalam surat pengantar tersebut?
