# Cherry Leaves Mildew Detection

## Introduction
Farmy & Foods have a mildew problem. Their cherry plantation operations have a growing powdery mildew issue. Powdery mildew, a fungal disease that affects many plant species, typically starts off as small white circular spots on the surface of leaves, and as the mildew progresses the spots may expand to cover the entire surface of each leaf. Infected leaves may become distorted, curl and fall from the tree. The disease deprives plants of water and nutrients, impacting their growth, blooming and development. 

The current system of manual inspection can take 30 minutes per tree, plus the time to apply treatment if necessary. Farmy & Foods' operation spans thousands of trees across multiple locations, rendering the manual process inadequate. 

With this in mind, a machine-learning system has been proposed to detect powdery mildew in cherry leaf images, greatly increasing the speed of detection. All that will be needed is a photograph of a leaf from a tree and for that image to be fed into the ML model to discern if powdery mildew is present.

The [dataset](https://www.kaggle.com/codeinstitute/cherry-leaves) input comprises of over 4200 cherry leaf images from Farmy & Foods, pre-labelled healthy [uninfected] or unhealthy [infected with powdery mildew]. 

## Business Case Assessment
### The business requirements are:
<ol><li> The client is interested in conducting a study to visually differentiate a cherry leaf that is healthy from one that is infected with powdery mildew.</li>
<ul><li>a. The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew based on visual analysis.</li>
<li>b. A 97% degree of accuracy has been agreed.</li></ul>
<li> The client requires a dashboard to view the results. </li>
<li> The client provided the data under an NDA (non-disclosure agreement), therefore the data should only be shared with professionals that are officially involved in the project, for ethical and privacy reasons.</li></ol>

### The benefit for the client is:
- Manual checking - shown to be unscalable for an operation of this magnitude - will be minimised.
- They will be able to ensure that the product they supply to the market is not compromised.
- The spread of powdery mildew can be controlled with early diagnosis and treatment of infected plants.

### The data analyst will therefore:
- Use inputs of cherry leaf images known to be either healthy or infected with mildew to train an image analysis model.
- Use known data to train the model to be able to predict whether unknown images are infected or not.
- Analysis will be conducted on both average and variability images for both healthy and unhealthy leaves.
- A binary classifier will be used to give a simple Healthy/Unhealthy output, with a calculation of the expected accuracy on each image. 
- There will be an image montage for each of the two classes.

## Hypothesis and Validation:
* List here your project hypothesis(es) and how you envision validating it (them).

## Dashboard design:
- A project summary page.
- A page listing the findings related to the study to visually differentiate healthy and mildew-infected leaves.
- A page containing a link to download cherry-leaf images for live prediction, and a user interface with a file uploader. The user must be able to upload multiple images, and for each image a prediction statement [healthy/unhealthy] plus the probability of correct prediction should be present. A downloadable table with the results must also be available.
- A project hypothesis page, with details of how it was validated.
- A more technical page for data analysts displayinng model performance. 


## Unfixed Bugs
* You will need to mention unfixed bugs and why they were unfixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a significant variable for consideration, paucity of time and difficulty understanding implementation is not a valid reason to leave bugs unfixed.

## Deployment
### Heroku

* The App live link is: https://YOUR_APP_NAME.herokuapp.com/ 
* Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
* The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create a new App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select the repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy (typically Main), then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click now the button Open App on the top of the page to access your App.
6. If the slug size is too large then add large files not required for the app to the .slugignore file. 

## Main Data Analysis and Machine Learning Libraries
* Here you should list the libraries used in the project and provide an example(s) of how you used these libraries.

## Credits 

* In this section, you need to reference where you got your content, media and from where you got extra help. It is common practice to use code from other repositories and tutorials. However, it is necessary to be very specific about these sources to avoid plagiarism. 
* You can break the credits section up into Content and Media, depending on what you have included in your project. 

### Content 

- The text for the Home page was taken from Wikipedia Article A.
- Instructions on how to implement form validation on the Sign-Up page were taken from [Specific YouTube Tutorial](https://www.youtube.com/).
- The icons in the footer were taken from [Font Awesome](https://fontawesome.com/).

### Media

- The photos used on the home and sign-up page are from This Open-Source site.
- The images used for the gallery page were taken from this other open-source site.


## Acknowledgements (optional)
* Thank the people that provided support throughout this project.





n.b.
## ML Business Case
* In the previous bullet, you potentially visualised an ML task to answer a business requirement. You should frame the business case using the method we covered in the course.

- need to check course materials






no img augmentation as 2100 each is enough(?) ~ do I need augmentation??
image size is 256,256. 4000+ imgs at this size is too big for a regular heroku deployment, so images are resized to 80,80 (I haven't actually done this... got to work out how exactly)
av. img is a black image. can I edit this to make it better..?
in data_visualisation I've copied Diff and Montage functions directly. Edit to make them more personalised
We're using a CNN model (convolutional neural network), binary classifier
Think: CRISP-DM, AGILE - justify decisions made using these frameworks

quote from video (re: base CI model for malaria detection): Convolution layers are used to select the dominant pixel value
from the non-dominant pixels in images using filters whereas the max-pooling layers reduce
the size of image by extracting only the dominant pixels within the pool size. This reduced image
only contains the dominant feature of the image. The combination of these two layers removes
the nonessential part of the image and reduces complexity, therefore providing more accuracy.
Here, in our model, we have used filters of a 3 by 3 matrix and their numbers are varied from
32 to 64, the pool size for the max-pooling layer is a 2 by 2 matrix. The input image shape in the
first convolution layer is served as the average image shape from the data visualization video.
As well as this, the flatten layer is used to flatten the matrix into a vector,
which means a single list of all values, and to
feed it into a dense layer. The dense layer then does the mathematical operation
and gives the output. In our model, we have used 128 nodes in one of our Dense layers,
and 1 node in the output dense layer, because we want single output at a time using the sigmoid
activation function. This output activation function defines the probabilistic result.
Finally, the Dropout layer is used in this network to drop 50 percent of the nodes
to avoid overfitting the model.
We use a combination of all these layers to develop a convolution neural network,
which we will use in the final prediction tasks.
The activation function in the output layer is sigmoid. The loss, optimizer,
and metrics used in the model compiler are binary cross-entropy, adam, and accuracy respectively.
These three hyperparameters along with the activation function of the output layer
which control the classification and regression problems.

tweaking: relu, adam, kernel_size, filters
> For binary classification, we use binary cross-entropy as the loss function,
*adam, rmspro, SGD or any other function as an optimizer* and accuracy as metrics, along
with sigmoid as an activation function in the output layer (1 node instead of the 2 needed for softmax).

Hyperparameters tune the model parameters so that we can control the performance of the model.
In our model, we have used dropout, loss, optimizer, and nodes in the dense layer as
the hyperparameter that can be tuned to control the model performance. However, *the selection
of the optimum hyperparameter value is a trial and error type method. Hence, we use different model
tuning functions present in the Keras packages.* I recommend reading more about hyperparameter
tuning in TensorFlow and Keras on your own and trying to tune models later to see their effect.
One more piece of information about convolution neural network performance tuning is that
*we use transfer learning techniques which are trained layers of neural networks. Read more
about transfer learning on Tensorflow and Keras documentation to learn more about this technology*.

------------------

![CI logo](https://codeinstitute.s3.amazonaws.com/fullstack/ci_logo_small.png)

## Codeanywhere Template Instructions

Welcome,

This is the Code Institute student template for Codeanywhere. We have preinstalled all of the tools you need to get started. It's perfectly ok to use this template as the basis for your project submissions. Click the `Use this template` button above to get started.

You can safely delete the Codeanywhere Template Instructions section of this README.md file,  and modify the remaining paragraphs for your own project. Please do read the Codeanywhere Template Instructions at least once, though! It contains some important information about the IDE and the extensions we use. 

## How to use this repo

1. Use this template to create your GitHub project repo

1. Log into <a href="https://app.codeanywhere.com/" target="_blank" rel="noreferrer">CodeAnywhere</a> with your GitHub account.

1. On your Dashboard, click on the New Workspace button

1. Paste in the URL you copied from GitHub earlier

1. Click Create

1. Wait for the workspace to open. This can take a few minutes.

1. Open a new terminal and <code>pip3 install -r requirements.txt</code>

1. In the terminal type <code>pip3 install jupyter</code>

1. In the terminal type <code>jupyter notebook --NotebookApp.token='' --NotebookApp.password=''</code> to start the jupyter server.

1. Open port 8888 preview or browser

1. Open the jupyter_notebooks directory in the jupyter webpage that has opened and click on the notebook you want to open.

1. Click the button Not Trusted and choose Trust.

Note that the kernel says Python 3. It inherits from the workspace so it will be Python-3.8.12 as installed by our template. To confirm this you can use <code>! python --version</code> in a notebook code cell.


## Cloud IDE Reminders

To log into the Heroku toolbelt CLI:

1. Log in to your Heroku account and go to *Account Settings* in the menu under your avatar.
2. Scroll down to the *API Key* and click *Reveal*
3. Copy the key
4. In the terminal, run `heroku_config`
5. Paste in your API key when asked

You can now use the `heroku` CLI program - try running `heroku apps` to confirm it works. This API key is unique and private to you, so do not share it. If you accidentally make it public then you can create a new one with _Regenerate API Key_.


## Dataset Content
* The dataset is sourced from [Kaggle](https://www.kaggle.com/codeinstitute/cherry-leaves). We then created a fictitious user story where predictive analytics can be applied in a real project in the workplace.
* The dataset contains +4 thousand images taken from the client's crop fields. The images show healthy cherry leaves and cherry leaves that have powdery mildew, a fungal disease that affects many plant species. The cherry plantation crop is one of the finest products in their portfolio, and the company is concerned about supplying the market with a compromised quality product.



## Business Requirements
The cherry plantation crop from Farmy & Foods is facing a challenge where their cherry plantations have been presenting powdery mildew. Currently, the process is manual verification if a given cherry tree contains powdery mildew. An employee spends around 30 minutes in each tree, taking a few samples of tree leaves and verifying visually if the leaf tree is healthy or has powdery mildew. If there is powdery mildew, the employee applies a specific compound to kill the fungus. The time spent applying this compound is 1 minute.  The company has thousands of cherry trees, located on multiple farms across the country. As a result, this manual process is not scalable due to the time spent in the manual process inspection.

To save time in this process, the IT team suggested an ML system that detects instantly, using a leaf tree image, if it is healthy or has powdery mildew. A similar manual process is in place for other crops for detecting pests, and if this initiative is successful, there is a realistic chance to replicate this project for all other crops. The dataset is a collection of cherry leaf images provided by Farmy & Foods, taken from their crops.


* 1 - The client is interested in conducting a study to visually differentiate a healthy cherry leaf from one with powdery mildew.
* 2 - The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew.


## Hypothesis and how to validate?
* List here your project hypothesis(es) and how you envision validating it (them).


## The rationale to map the business requirements to the Data Visualisations and ML tasks
* List your business requirements and a rationale to map them to the Data Visualisations and ML tasks.


## ML Business Case
* In the previous bullet, you potentially visualised an ML task to answer a business requirement. You should frame the business case using the method we covered in the course.


## Dashboard Design
* List all dashboard pages and their content, either blocks of information or widgets, like buttons, checkboxes, images, or any other items, that your dashboard library supports.
* Finally, during the project development, you may revisit your dashboard plan to update a given feature (for example, at the beginning of the project, you were confident you would use a given plot to display an insight, but later, you chose another plot type).


## Unfixed Bugs
* You will need to mention unfixed bugs and why they were unfixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a significant variable for consideration, paucity of time and difficulty understanding implementation is not a valid reason to leave bugs unfixed.

## Deployment
### Heroku

* The App live link is: https://YOUR_APP_NAME.herokuapp.com/ 
* Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
* The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click now the button Open App on the top of the page to access your App.
6. If the slug size is too large then add large files not required for the app to the .slugignore file. 


## Main Data Analysis and Machine Learning Libraries
* Here you should list the libraries used in the project and provide an example(s) of how you used these libraries.


## Credits 

* In this section, you need to reference where you got your content, media and from where you got extra help. It is common practice to use code from other repositories and tutorials. However, it is necessary to be very specific about these sources to avoid plagiarism. 
* You can break the credits section up into Content and Media, depending on what you have included in your project. 

### Content 

- The text for the Home page was taken from Wikipedia Article A.
- Instructions on how to implement form validation on the Sign-Up page were taken from [Specific YouTube Tutorial](https://www.youtube.com/).
- The icons in the footer were taken from [Font Awesome](https://fontawesome.com/).

### Media

- The photos used on the home and sign-up page are from This Open-Source site.
- The images used for the gallery page were taken from this other open-source site.



## Acknowledgements (optional)
* Thank the people that provided support throughout this project.
