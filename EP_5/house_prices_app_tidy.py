import pandas as pd
import streamlit as st
import numpy as np
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from utilils import load_house_attributes, process_house_attributes, load_house_images
from utilils import montage_image, load_it, train_model
import plotly.express as px
import keras.backend.tensorflow_backend as tb
import cv2
import glob
tb._SYMBOLIC_SCOPE.value = True




def main():
    st.title('Mixed data to Predict House Prices')
# I am laoding the data here so that it can be avialable to all pages
    page = st.sidebar.selectbox("Choose a page", ['Exploration', 'Modelling', 'Prediction'])
    Path_to_data = "/Users/abusnina/Documents/Training/Streamlit/house_prices_demo/keras-regression/Houses-dataset/Houses Dataset/HousesInfo.txt"
    data = load_house_attributes(Path_to_data)
    Path_to_images = "/Users/abusnina/Documents/Training/Streamlit/house_prices_demo/keras-regression/Houses-dataset/Houses Dataset"

    if st.button("Load Images?"):
        images, Path_display = load_house_images(data, Path_to_images)
        images = images / 255.0
        st.success("Images Loaded successfully!")



    if page == 'Exploration':

        uploaded_file = st.file_uploader("Choose a TXT file", type="txt")
        if uploaded_file is not None:
            data = load_house_attributes(uploaded_file)
        # Visualize some stuff
            x_axis = st.radio("Select X-Axis", data.columns.tolist())
            fig = px.scatter(data, x=x_axis, y="price")
            fig
            x_freq = st.radio("Select Varaible", data.columns.tolist())
            fig_bar = px.histogram(data, x=x_freq )
            fig_bar
            x_color = st.radio("Select Coloring Varaible", data.columns.tolist())
            fig = px.scatter_matrix(data,
            dimensions=data.columns.tolist(),
            color=x_color)
            fig

        if st.checkbox("Show Data?"):
            rows = st.number_input("How many Rows to Show?", min_value=1, max_value = data.shape[0])
            st.dataframe(data.head(rows))

        house_no  = st.number_input("Choose House No:", min_value=1, max_value =  data.shape[0])

        if st.checkbox("Show House Image?"):
           _, Path_display = load_house_images(data, Path_to_images)
           st.image( Path_display[house_no], width = 164)





        if st.checkbox("Show Montaged Image?"):
            images, _ = load_house_images(data, Path_to_images)
            st.image(images[house_no ], channels="BGR", width = 512)

#*******************************************************************************
#************************************* TRAIN IT *******************************
#*******************************************************************************
    elif page == 'Modelling':
        st.markdown("<h1 style='text-align: center; color: balck;'>Build Your Model </h1>", unsafe_allow_html=True)

        #st.title('Build Your Model')

        test_size = st.slider("Splitting percentage", 0.10, 0.30, 0.25)
        #st.write("The size of the test set is : ", testAttrX.shape[0])
        no_kernel = st.sidebar.number_input("No. Kernels",min_value =1 , step =1,)
        struct_list = []
        for i in range(no_kernel):
            filter_size = st.sidebar.slider("Size of Kernel {}".format(i), min_value=6, step=2)
            struct_list.append(filter_size)
        opt = st.sidebar.selectbox("Select Optimizer", ["adam", "sgd", "ARMSprop", "Adagrad"])

        no_epochs = st.sidebar.number_input("No. Epochs",min_value =3 ,max_value = 20 , step =1,)

        if len(struct_list) >= 1 and st.sidebar.button("Train Model"):
            images, Path_display = load_house_images(data, Path_to_images)
            images = images / 255.0
            model, history, labelizer, cs, maxPrice = train_model(struct_list, opt, no_epochs, test_size, data, images)

            st.success('Training is done!')

            st.write("Loss of the model:")

            fig_loss = px.line()
            fig_loss.add_scatter(x=history.index, y=history.loss, name= "loss" )
            fig_loss.add_scatter(x=history.index, y=history.val_loss, name= "val_loss" )
            fig_loss

            st.write("Accuracy of the model:")

            fig_acc = px.line()
            fig_acc.add_scatter(x=history.index, y=history.accuracy, name= "accuracy" )
            fig_acc.add_scatter(x=history.index, y=history.val_accuracy, name= "val_accuracy" )
            fig_acc

            st.markdown("<h3 style='text-align: left; color: darkred;'>Training and Validation Performance - Last 3 Epochs </h3>", unsafe_allow_html=True)

            st.write(history.tail(3))


        if st.sidebar.button("Save Model"):
            model, history, labelizer, cs, out_scaler = train_model(struct_list, opt, no_epochs, test_size, data, images)
            model.save('my_model.h5')
            pickle.dump(cs, open('cs.pkl', 'wb'))
            pickle.dump(out_scaler, open('out_scaler.pkl', 'wb'))
            pickle.dump(labelizer, open('labelizer.pkl', 'wb'))
            st.success('Model Saved! Enjoy it :')


#*******************************************************************************
#************************************ USE THE MODEL ****************************
#*******************************************************************************
    else:
        st.title('Use the Model')

        model_path = st.file_uploader("Choose a h5 file", type="h5")
        binaizer_path = st.file_uploader("Choose a Binazer file", type="pkl")
        scaler_path = st.file_uploader("Choose a scaler file", type="pkl")

        bdr = st.slider("No of Bedrooms: ", int(data["bedrooms"].min()),  int(data["bedrooms"].max()), step=1 )
        bath = st.slider("No of Bathrooms: ", int(data["bathrooms"].min()), int(data["bathrooms"].max()),   step=1)
        area = st.slider("Area of the house: ", float(data["area"].min()), float(data["area"].max()))
        zip = st.slider("Where is the house: ", int(data["zipcode"].min()), int(data["zipcode"].max()) )

        df_pred = pd.DataFrame({"bedrooms": bdr, "bathrooms":bath,  "area":area, "zipcode":zip}, index=[0])
        #Upload Images of the house
        st.subheader("Choose how the house looks like: ")
        bath_img = st.file_uploader("Choose bathroom image", type="jpg")
        bed_img = st.file_uploader("Choose bedroom image", type="jpg")
        kitch_img = st.file_uploader("Choose kitchen image", type="jpg")
        front_img = st.file_uploader("Choose front image", type="jpg")


        if  bath_img and bed_img and kitch_img and  front_img is not None:
            montaged_img , Front = montage_image([bath_img, bed_img, kitch_img, front_img])

        show_it = st.checkbox("Show Montaged Image?")
        if show_it:
           st.image(montaged_img, channels="BGR", width = 256)



        if st.button("Predict"):
            montaged_img , Front = montage_image([bath_img, bed_img, kitch_img, front_img])
            montaged_img = montaged_img / 255.0
            model, labelizer, cs, out_scaler = load_it()
            if  labelizer and cs is not None:
                zip_cat = labelizer.transform(df_pred["zipcode"])
                continus_var = cs.transform(df_pred.loc[:, ["bedrooms", "bathrooms", "area"]])
                attribX = np.hstack([zip_cat, continus_var])
                pred_price =  model.predict([attribX, montaged_img])   #.flatten()[0],1)
                pred_price = out_scaler.inverse_transform(pred_price).flatten()[0]
                st.header("The predicted price based on these inputs is: ")
                font = cv2.FONT_HERSHEY_SIMPLEX
                Front = cv2.resize(Front, (256, 256))
                st.image(cv2.putText(Front, "$" + str(pred_price), (15,180), font, 1, (0, 255, 0), 2, cv2.LINE_AA))







main()
