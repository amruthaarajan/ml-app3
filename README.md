# ML Api for linear regression

## Procedure
1. Start a virtual environment and install requirements
2. Run the API


## File Structure
* app_name
  * app.py: Flask API application
  * requirements.txt: list of packages that the app will import
  * temp: directory for storing png files


## Run the API
1. Run the Flask API locally for testing. Go to directory with `app.py`.

```bash
python app.py
```


## Appendix

### Virtual Environment
1. Create new virtual environment
```bash
cd ~/.virtualenvs
virtualenv name-of-env
```
2. Activate virtual environment
```
source env/bin/activate
```
3. Go to app.py directory where `requirements.txt` is also located
4. Install required packages from `requirements.txt`
```bash
pip install -r requirements.txt
```
You will only have to install the `requirements.txt` when working with a new virtual environment.
