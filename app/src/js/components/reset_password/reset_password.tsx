import Button from '@material-ui/core/Button'
import CssBaseline from '@material-ui/core/CssBaseline'
import { withStyles } from '@material-ui/core/styles'
import { Field, Form, Formik, FormikHelpers } from 'formik'
import { TextField } from 'formik-material-ui'
import React from 'react'
import { loginButtonStyle, loginStyle, loginTextFieldStyle } from '../../styles/login'

interface Values {
  /** password field value */
  password: string,
  /** password confirm field value */
  password1: string
}

export interface ClassType {
  /** body */
  body: string,
  /** form */
  form: string,
}

export interface Props {
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
            password1: ''
          }}
          validate={this.validate}
          onSubmit={this.submit}
          >
          {({ submitForm, isSubmitting }) => (
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
          </Form>)}
        </Formik>
      </div>
    )
  }

  /** field validation */
  private validate = (values: Values) => {
    const errors: Partial<Values> = {}
    if (!values.password) {
      errors.password = 'Required'
    } else if (values.password.length < 6) {
      errors.password = 'At least 6 characters in length'
    }
    if (!values.password1) {
      errors.password1 = 'Required'
    } else if (values.password1.length < 6) {
      errors.password1 = 'At least 6 characters in length'
    }
    if (values.password !== values.password1 && !errors.password1) {
      errors.password1 = 'Password unmatch'
    }
    return errors
  }

  /** submit the form */
  private submit = (values: Values, helper: FormikHelpers<Values>) => {
    setTimeout(() => {
      helper.setSubmitting(false)
      alert(JSON.stringify(values, null, 2))
    }, 500)
  }
}

const LoginText = withStyles(loginTextFieldStyle)(TextField)
const LoginButton = withStyles(loginButtonStyle)(Button)

/** export Create page */
export default withStyles(loginStyle, { withTheme: true })(ResetPassword)
