# PredictInterface
A flask based Deployment for Machine learning algos.
This app is aimed to produce risk assesment predictions for patients that underwent liver resection.
![image](https://github.com/AlaouiMdaghriAhmed/PredictInterface/assets/77700915/c8665a0d-0bdd-48ac-baaa-a24441e79c19)
![image](https://github.com/AlaouiMdaghriAhmed/PredictInterface/assets/77700915/67c3e6f3-ccea-4cc4-9c26-aac107f1286c)

## How To Run
1. Install `virtualenv`:
```
$ pip install virtualenv
```

2. Open a terminal in the project root directory and run:
```
$ virtualenv env
```

3. Then run the command:
```
$ .\env\Scripts\activate
```

4. Then install the dependencies:
```
$ (env) pip install -r requirements.txt
```

5. Finally start the web server:
```
$ (env) python app.py
```

This server will start on port 5000 by default. You can change this in `app.py` by changing the following line to this:

```python
if __name__ == "__main__":
    app.run(debug=True, port=<desired port>)
```

