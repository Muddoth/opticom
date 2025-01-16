# opticom
a mashup of eye tracking and aac board front end for non-verbal communication
1. SETUP

    Virtual Environment
        write python -m venv myenv in terminal
        write myenv/Scripts/activate in terminal

2. FOCUS FILES
    salaman.py 
        a simple gaze controlled cursor program without utilising an ANN like MLP

    Opticom/Gazer.py    
        implements MLP
        if it doesn't detect a training data file(e.g. calibration_data.pkl), you may start calibration

        CALIBRATION
           while the window of your face is in view,
            use the mouse and place the cursor at multiple positions on the screen
            for each position, keep your head still, and look at the mouse, then press 'c'
            in the terminal, the coordinates of your iris and mouse will be displayed
            these are the same data being appended to .pkl file that will be used during training
            repeat for as many times you want
            once youve decided to stop the calibration, you may start the training process by pressing 't'

---------------
3. WEB APPLICATION FRAMEWORK
    FOLDER : OptiCom
        uses a flask framework and im still figuring out how to allow proper communication of front end with the back end 
        since now the iris and mouse coordinates will be received from the front end
        
        1. base.html - is the landing page / navigates to signup.html and login.html
        2. home - is user's main page after registering / logging in
        3. calibrate.html - where the user can recalibrate the gaze-controlled navigation
        4. history.html - is where the user can access pass selected words
        5. app.py - runs the web application in localhost
        6. Opticom/opticom/gazer.py - the backend for eye tracking logic
        
