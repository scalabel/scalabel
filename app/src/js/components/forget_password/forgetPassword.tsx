import Button from '@material-ui/core/Button'
import CssBaseline from '@material-ui/core/CssBaseline'
import { withStyles } from '@material-ui/core/styles'
import { Validator } from 'class-validator'
import { Field, Form, Formik, FormikHelpers } from 'formik'
import { TextField } from 'formik-material-ui'
import React from 'react'
import { loginButtonStyle, loginStyle, loginTextFieldStyle } from '../../styles/login'

interface Values {
  /** email field value */
  email: string,
  /** server side error */
  general: string
}

interface ClassType {
  /** body */
  body: string,
  /** form */
  form: string,
}

interface Props {
  /** Create classes */
  classes: ClassType
}

/**
 * Component which display the create page
 * @param {object} props
 * @return component
 */
class ForgetPassword extends React.Component<Props> {

  /**
   * renders the create page
   * @return component
   */
  public render () {
    return (
      <div className={this.props.classes.body}>
        <CssBaseline />
        <Formik
          initialValues={{
            email: '',
            general: ''
          }}
          validate={this.validate}
          onSubmit={this.submit}
          >
          {({ submitForm, isSubmitting, errors }) => (
          <Form className={this.props.classes.form}>
            <Field
              component={LoginText}
              margin='normal'
              required
              fullWidth
              id='email'
              label='Email Address'
              name='email'
              autoComplete='email'
              autoFocus
            />
            <LoginButton
              type='submit'
              disabled={isSubmitting}
              onClick={submitForm}
              fullWidth
            >
              Send Recovery Email
            </LoginButton>
            <div style={{ color: 'red' }}>{errors.general}</div>
          </Form>)}
        </Formik>
      </div>
    )
  }

  /** field validation */
  private validate = (values: Values) => {
    const errors: Partial<Values> = {}
    const validator = new Validator()
    if (validator.isEmpty(values.email)) {
      errors.email = 'Required'
    } else if (!validator.isEmail(values.email)) {
      errors.email = 'Invalid email address'
    }
    return errors
  }

  /** submit the form */
  private submit = (values: Values, helper: FormikHelpers<Values>) => {
    fetch('/api/auth/forget_password', {
      method: 'post',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(values)
    })
    .then((response) => response.json())
    .then((data) => {
      helper.setSubmitting(false)
      if (data.code === 200) {
        // TODO: Stay and popup a dialog
        location.href = `/reset_password?token=${data.data.token}`
      } else {
        helper.setFieldError('general', data.message)
      }
    })
    .catch(() => helper.setSubmitting(false))
  }
}

const LoginText = withStyles(loginTextFieldStyle)(TextField)
const LoginButton = withStyles(loginButtonStyle)(Button)

/** export Create page */
export default withStyles(loginStyle, { withTheme: true })(ForgetPassword)
