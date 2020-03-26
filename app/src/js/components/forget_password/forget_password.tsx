import Button from '@material-ui/core/Button'
import CssBaseline from '@material-ui/core/CssBaseline'
import { withStyles } from '@material-ui/core/styles'
import { Field, Form, Formik, FormikHelpers } from 'formik'
import { TextField } from 'formik-material-ui'
import React from 'react'
import { loginButtonStyle, loginStyle, loginTextFieldStyle } from '../../styles/login'

interface Values {
  /** email field value */
  email: string
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
            email: ''
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
          </Form>)}
        </Formik>
      </div>
    )
  }

  /** field validation */
  private validate = (values: Values) => {
    const errors: Partial<Values> = {}
    if (!values.email) {
      errors.email = 'Required'
    } else if (
      !/^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}$/i.test(values.email)
    ) {
      errors.email = 'Invalid email address'
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
export default withStyles(loginStyle, { withTheme: true })(ForgetPassword)
