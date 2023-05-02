# CMPUT 466/566 Final Project Report

# Author: Leen Alzebdeh

    ## Introduction

    I will attempt to predict 4 attributes of Edmonton's weather - max temperature in C, min temperature in C, mean temperature in C, and total precipitation in meters - based on six features: day of the month, month of the year, weather monitoring station's latitude, station's longitude and station's elevation in meters.

    I utilized the dataset of daily weather in Edmonton, found \href{https://data.edmonton.ca/Environmental-Services/Weather-Data-Daily-Environment-Canada/s4ws-tdws}{here}. There are $\sim$ 71.6k entries.

    I will use linear regression, neural net regression and support vector machine (SMV) as the machine learning algorithms. I will use a training-validation-test split with hyper-parameter tuning. I used risk in the form of mean absolute error to evaluate the performance of the model.


    ## Methods

    ### Linear Regression

        For the training loss/ objective I used mean square error (MSE). I manually defined the linear regression model but used sklearn.metrics for the mean absolute error (MAE). I used normalized data and targets for training (but denormalized it for risk). I normalized inputs using the formula: $\tilde{z} = \frac{z - mean(z)}{std(z)}$.

        I fixed parameters at a batch size of 32 and 100 epochs. I experimented with learning parameter decay, L1 and L2 regularization.
        For the learning parameter's tuning, I loop over the set \{1e-1, 1e-2, 1e-3, 1e-4\} and for the $\lambda$ for regularization, I loop over the set \{0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6\}.
        For learning parameter decay, I used step-decay, where I halve the rate every 10 epochs.

    ### Neural Net Regression

        While linear regression is limited to only learn the linear relationship between the features and targets. To better model the problem, we can learn the non-linear relationship between the features and target, using neural networks, which utilize a non-linear activation function in each layer. we are in need of other techniques.

        I fixed parameters at a batch size of 64, 50 epochs. I use three densely connected layers, with ReLu activation and a dropout. The last layer is a dense layer with output unit size of 4. I tune the dropout rate by looping over the set \{0.3, 0.4, 0.5, 0.6, 0.7, 0.8\} and I experiment with different weight initialization kernels: random, normal Gaussian and He uniform initialization, in addition to two optimizers: Adam with learning rate 0.01, and a RMSProp.

    ### Support Vector Machine (SVM)

        Another method I experimented with is SVM, as it works effectively in cases where we have easily separable classes and is generally more memory efficient. To adjust for multiple class output, I used MultiOutputRegressor from sklearn, which fits four regressors for each class.

        I experimented with different kernels to determine the best fitting one.

    ## Results


    ### Linear Regression

    The results of tuning concluded the best model is with no L1 or L2 reguralization and no decay. I found the best test risk to be 7.654087.

    ### Regularization

    I fixed alpha, then for each alpha in the set, I fix a $/lambda$ then get the test risk. I found the best result, at 7.654087, to come from having no regularization term.


    \item For L1 regularization, I found a learning rate (alpha) of 0.0001 and a $\lambda$ of 0.1 to produce the lowest test risk of 7.775. Below are a few figures from L1 tuning.
    \item For L2 regularization, I found a learning rate of 0.0001 and a $\lambda$ of 0 (no L2 term) to produce the lowest test risk of 7.654087. Below are a few figures from L2 tuning.
    \end{enumerate}
    \begin{figure}[H]%
        \centering
        \subfloat[\centering The learning curve of the training MSE (alpha = 0.01, $\lambda$ = 0.0001)]{{\includegraphics[width=3.5cm]{linear regression/l1/losses_train_l1_lr_0.01_lmd_0.0001.jpg} }}%
        \subfloat[\centering The learning curve of the validation MAE (alpha = 0.01, $\lambda$ = 0.0001)]{{\includegraphics[width=3.5cm]{linear regression/l1/valid_l1_lr_0.01_lmd_0.0001.jpg} }}%
        \qquad
        \subfloat[\centering The learning curve of the training MSE (alpha = 0.0001, $\lambda$ = 0.1)]{{\includegraphics[width=3.5cm]{linear regression/l1/losses_train_l1_lr_0.0001_lmd_0.1.jpg} }}%
        \subfloat[\centering The learning curve of the validation MAE (alpha = 0.0001, $\lambda$ = 0.1)]{{\includegraphics[width=3.5cm]{linear regression/l1/valid_l1_lr_0.0001_lmd_0.1.jpg} }}%
        \caption{Learning curve of training and validation for L1 Regularization}%
        \label{fig:example}%
    \end{figure}

    \begin{figure}[H]%
        \centering
        \subfloat[\centering The learning curve of the training MSE (alpha = 0.001, $\lambda$ = 0.1)]{{\includegraphics[width=3.5cm]{linear regression/l2/losses_train_l2_lr_0.001_lmd_0.1.jpg} }}%
        \subfloat[\centering The learning curve of the validation risk (alpha = 0.001, $\lambda$ = 0.1)]{{\includegraphics[width=3.5cm]{linear regression/l2/valid_l2_lr_0.001_lmd_0.1.jpg} }}%
        \qquad
        \subfloat[\centering The learning curve of the training mean MSE (alpha = 0.1, $\lambda$ = 0)]{{\includegraphics[width=3.5cm]{linear regression/l2/losses_train_l2_lr_0.1_lmd_0.jpg} }}%
        \subfloat[\centering The learning curve of the validation MAE (alpha = 0.1, $\lambda$ = 0)]{{\includegraphics[width=3.5cm]{linear regression/l2/valid_l2_lr_0.1_lmd_0.jpg} }}%
        \caption{Learning curve of training and validation for L2 Regularization}%
        \label{fig:example}%
    \end{figure}

    \subsubsection*{Step Decay}

    I experimented with a few initial learning rates. After experimenting with different factors, I decided on a factor of 0.5 every 10 epochs. Thus every 10 epochs, the learning rate decreases by half.

    \begin{enumerate}
    \item I found an initial learning rate (alpha) of 0.1 to produce the lowest test risk of 7.6947923. Below are a few figures from decay tuning.
    \end{enumerate}
    \begin{figure}[H]%
        \centering
        \subfloat[\centering The learning curve of the training MSE (alpha = 0.001, $\lambda$ = 0.1)]{{\includegraphics[width=3.5cm]{linear regression/l2/losses_train_l2_lr_0.001_lmd_0.1.jpg} }}%
        \subfloat[\centering The learning curve of the validation risk (alpha = 0.001, $\lambda$ = 0.1)]{{\includegraphics[width=3.5cm]{linear regression/l2/losses_train_l2_lr_0.001_lmd_0.1.jpg} }}%
        \qquad
        \subfloat[\centering The learning curve of the training mean MSE (alpha = 0.1, $\lambda$ = 0)]{{\includegraphics[width=3.5cm]{linear regression/l2/losses_train_l2_lr_0.1_lmd_0.jpg} }}%
        \subfloat[\centering The learning curve of the validation MAE (alpha = 0.1, $\lambda$ = 0)]{{\includegraphics[width=3.5cm]{linear regression/l2/valid_l2_lr_0.1_lmd_0.jpg} }}%
        \caption{Learning curve of training and validation for learning rate step-decay}%
        \label{fig:example}%
    \end{figure}

    \subsection*{Neural Net Regression}

    After tuning I found the best model to have a dropout layer of 0.3 and a RMSProp optimizer with He uniform weight initializer kernel. The lowest test risk it produced was 5.9898529052734375.

    \subsubsection*{Optimizers}

    \begin{enumerate}

    \item For an Adam optimizer with an initial learning rate of 0.01, I found a test risk of 7.744905948638916. Below are figures from Adam optimizer.
    \item For an RMSProp optimizer (and He Uniform weight initializer kernel), I found the lowest test risk of 5.9898529052734375. Below are figures from RMSProp optimizer.
    \end{enumerate}
    \begin{figure}[H]%
        \centering
        \subfloat[\centering The learning curve of the training MSE (Adam optimizer)]{{\includegraphics[width=3.5cm]{nn/neural_network_train_adam_drop_0.3.jpg} }}%
        \subfloat[\centering The learning curve of the validation MAE (Adam optimizer)]{{\includegraphics[width=3.5cm]{nn/neural_network_val_adam_drop_0.3.jpg} }}%
        \qquad
        \subfloat[\centering The learning curve of the training MSE (RMSProp optimizer)]{{\includegraphics[width=3.5cm]{nn/neural_network_train_rms_drop_0.3.jpg} }}%
        \subfloat[\centering The learning curve of the validation MAE (RMSProp optimizer)]{{\includegraphics[width=3.5cm]{nn/neural_network_val_rms_0.3.jpg} }}%
        \caption{Learning curve of training and validation for Adam and RMSProp optimzers}%
        \label{fig:example}%
    \end{figure}

    \subsubsection*{Weight Initializer Kernels}

    \begin{enumerate}

    \item For a normal, Gaussian weight initialization kernel, I found a test risk of 7.744905948638916. Below are learning curves.
    \item For HE Uniform weight initialization kernel, I found the test risk of 5.9898529052734375. Figures can be found in figure 4.0 for the RMSProp optimizer.
    \end{enumerate}
    \begin{figure}[H]%
        \centering
        \subfloat[\centering The learning curve of the training MSE (normal kernel)]{{\includegraphics[width=3.5cm]{nn/neural_network_train_adam_drop_0.3.jpg} }}%
        \subfloat[\centering The learning curve of the validation MAE (normal kernel)]{{\includegraphics[width=3.5cm]{nn/neural_network_val_adam_drop_0.3.jpg} }}%
        \caption{Learning curve of training and validation for normal weight initialization}%
        \label{fig:example}%
    \end{figure}

    \subsubsection*{Dropout}

    \begin{enumerate}

    \item I found a dropout rate of 0.3 to produce the lowest test risk of . Figure for rate 0.3 can be found in figure 4.0 for RMSProp optimizer. Below a few figures of tuning canbe found.
    \end{enumerate}
    \begin{figure}[H]%
        \centering
        \subfloat[\centering The learning curve of the training MSE (dropout 0.6)]{{\includegraphics[width=3.5cm]{nn/neural_network_train_drop_0.6.jpg} }}%
        \subfloat[\centering The learning curve of the validation MAE (dropout 0.6)]{{\includegraphics[width=3.5cm]{nn/neural_network_val_drop_0.6.jpg} }}%
        \caption{Learning curve of training and validation for a dropout rate of 0.6}%
        \label{fig:example}%
    \end{figure}

    \subsection*{SVM Machine}
    I found linear kernel to produce the lowest test risk of 7.778022035132583.
    \section*{Conclusion}

    After experimenting with the three different machine learning models, I found using a neural network to produce the lowest risk I am able to get of 5.9898529052734375. I found the neural net to be the most memory intensive while the SVM was the least

    \section*{References}

    R, Srivignesh. “A Walk-through of Regression Analysis Using Artificial Neural Networks in Tensorflow.” Analytics Vidhya, August 16, 2021, \url{https://www.analyticsvidhya.com/blog/2021/08/a-walk-through-of-regression-analysis-using-artificial-neural-networks-in-tensorflow/}.

    “Training and evaluation with the built-in methods.” TensorFlow, Jan 10, 2022, \url{https://www.tensorflow.org/guide/keras/train_and_evaluate}.

    “Machine Learning Models.” MathWorks, \url{https://www.mathworks.com/discovery/machine-learning-models}.

    R, Srivignesh. “A Walk-through of Regression Analysis Using Artificial Neural Networks in Tensorflow.” Analytics Vidhya, March 27, 2021, \url{https://www.analyticsvidhya.com/blog/2020/03/support-vector-regression-tutorial-for-machine-learning/}.

    Versloot, Christian. “How To Perform Multioutput Regression With Svms In Python.” Feb 15, 2022, \url{https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-perform-multioutput-regression-with-svms-in-python.md}.

\end{document}
