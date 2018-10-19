from wtforms import Form, DecimalField, SubmitField
from wtforms.validators import DataRequired


class TheForm(Form):
    param1 = DecimalField(label='Sepal Length (cm):', places=2, validators=[DataRequired()])
    param2 = DecimalField(label='Sepal Width (cm):', places=2, validators=[DataRequired()])
    param3 = DecimalField(label='Petal Length (cm):', places=2, validators=[DataRequired()])
    param4 = DecimalField(label='Petal Width (cm):', places=2, validators=[DataRequired()])
    submit = SubmitField('Submit')
