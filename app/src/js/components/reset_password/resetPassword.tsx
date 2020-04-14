import Button from '@material-ui/core/Button'
import CssBaseline from '@material-ui/core/CssBaseline'
import { withStyles } from '@material-ui/core/styles'
import { Validator } from 'class-validator'
import { Field, Form, Formik, FormikHelpers } from 'formik'
import { TextField } from 'formik-material-ui'
import React from 'react'
import { loginButtonStyle, loginStyle, loginTextFieldStyle } from '../../styles/login'

interface Values {
  /** password field value */
  password: string,
  /** password confirm field value */
  password1: string,
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
class ResetPassword extends React.Component<Props> {

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
            password: '',
            password1: '',
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
              name='password'
              label='Password'
              type='password'
              id='password'
            />
            <Field
              component={LoginText}
              margin='normal'
              required
              fullWidth
              name='password1'
              label='Password Confirmation'
              type='password'
              id='password1'
            />
            <LoginButton
              type='submit'
              disabled={isSubmitting}
              onClick={submitForm}
              fullWidth
            >
              Reset Password
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
    if (validator.isEmpty(values.password)) {
      errors.password = 'Required'
    } else if (validator.maxLength(values.password, 6)) {
      errors.password = 'At least 6 characters in length'
    }
    if (validator.isEmpty(values.password1)) {
      errors.password1 = 'Required'
    } else if (validator.maxLength(values.password1, 6)) {
      errors.password1 = 'At least 6 characters in length'
    }
    if (validator.notEquals(values.password, values.password1)
      && !errors.password1) {
      errors.password1 = 'Password unmatch'
    }

    return errors
  }

  /** submit the form */
  private submit = (values: Values, helper: FormikHelpers<Values>) => {
    const result = new Map()
    const queryParams = window.location.search.substr(1).split('&amp;')
    queryParams.forEach((queryParam) => {
      const item = queryParam.split('=')
      result.set(item[0], decodeURIComponent(item[1]))
    })
    fetch('/api/auth/reset_password', {
      method: 'post',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ...values,
        token: result.get('token')
      })
    })
    .then((response) => response.json())
    .then((data) => {
      helper.setSubmitting(false)
      if (data.code === 200) {
        // TODO: Could change the following tips to a Dialog
        helper.setFieldError('general', 'Success. Redirecting to login page...')
        setTimeout(() => {
          location.href = `/login`
        }, 2000)
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
export default withStyles(loginStyle, { withTheme: true })(ResetPassword)
