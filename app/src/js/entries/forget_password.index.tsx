import { MuiThemeProvider } from '@material-ui/core/styles'
import React from 'react'
import ReactDOM from 'react-dom'
import ForgetPassword from '../components/forget_password/forgetPassword'
import { myTheme } from '../styles/theme'

ReactDOM.render(
  <MuiThemeProvider theme={myTheme}>
    <ForgetPassword />
  </MuiThemeProvider>
  , document.getElementById('forget-root'))
