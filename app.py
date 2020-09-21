import streamlit as st
from fastai.text import *
from fastai import *
import fastai
import matplotlib.cm as cm
from pathlib import Path, WindowsPath


# Title
st.title("Restaurant Sentiment App")

st.subheader(" I am 96% accuarte, excuse me for minimal misclassifications!")

status=st.radio('What would you like to do with your review?',('sentiment prediction','next word predicion'))

if status == 'sentiment prediction':
	st.subheader("Sentiment Prediction")
	review=st.text_area('Enter yout text here...')
else:
	st.subheader("Next Word Prediction")
	review=st.text_area('Enter yout text here...')
	n_words=st.number_input('Insert a number',min_value=1,max_value=300)







#import warnings

#warnings.filterwarnings('ignore')

#def predict(text,learner):
#    test = pd.DataFrame([text], columns=['text'])
#    learner.data.add_test(test['text'])
#    preds, y, losses = learner.get_preds(ds_type=DatasetType.Test, with_loss=True)
#    #print(preds.dtype,y.bool().dtype,losses.dtype)
#    classes = ['negative', 'positive']
#    #print(classes[preds.argmax(dim=1)[0].tolist()])
#    interp = TextClassificationInterpretation(learner, preds, y.bool(), losses)
#    result= interp.html_intrinsic_attention(text,cmap=cm.Purples)
#    return result
 
def nextwords(text,n_words=10):
#	path = WindowsPath("E:/MSc/GP/files/lm")
	learn = load_learner("./","lm_export.pkl")
	result= learn.predict(text, n_words)
	return result

if st.button("submit"):
	if status == 'sentiment prediction':
		learn=load_learner("./")
		predictions=learn.predict(review)
		if int(predictions[0])==1:
			st.success("positive sentiment")
		else:
			st.warning("negative sentiment")
	else:
		preds=nextwords(review,n_words)
		st.text(preds)


#csv = df.to_csv(index=False)
#b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
#href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
#st.markdown(href, unsafe_allow_html=True)



st.sidebar.header('About')
