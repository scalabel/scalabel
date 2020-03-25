import Button from '@material-ui/core/Button'
import CssBaseline from '@material-ui/core/CssBaseline'
import { withStyles } from '@material-ui/core/styles'
import TextField from '@material-ui/core/TextField'
import React from 'react'
import { loginButtonStyle, loginStyle, loginTextFieldStyle } from '../../styles/login'

export interface ClassType {
  /** body */
  body: string,
  /** img */
  img: string,
  /** break */
  break: string,
  /** link */
  link: string,
  /** intro */
  intro: string,
  /** form */
  form: string,
  /** checkbox */
  checkbox: string,
  /** bdd_font */
  bdd_font: string,
  /** btn */
  btn: string
}

export interface Props {
  /** Create classes */
  classes: ClassType
}

export interface State {
  /** boolean to force reload of the sidebar project list */
  open: boolean
}

/**
 * Component which display the create page
 * @param {object} props
 * @return component
 */
class ResetPassword extends React.Component<Props, State> {

  public constructor (props: Props) {
    super(props)
    this.state = {
      open: false
    }
  }

  /**
   * renders the create page
   * @return component
   */
  public render () {
    return (
      <div className={this.props.classes.body}>
        <CssBaseline />
        <form className={this.props.classes.form} noValidate>
          <LoginText
            margin='normal'
            required
            fullWidth
            name='password'
            label='Password'
            type='password'
            id='password'
          />
          <LoginText
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
            fullWidth
          >
            Reset Password
          </LoginButton>
        </form>
      </div>
    )
  }

}

const LoginText = withStyles(loginTextFieldStyle)(TextField)
const LoginButton = withStyles(loginButtonStyle)(Button)

/** export Create page */
export default withStyles(loginStyle, { withTheme: true })(ResetPassword)
