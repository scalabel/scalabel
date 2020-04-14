import { MuiThemeProvider } from '@material-ui/core/styles'
import React from 'react'
import ReactDOM from 'react-dom'
import ResetPassword from '../components/reset_password/resetPassword'
import { myTheme } from '../styles/theme'

ReactDOM.render(
  <MuiThemeProvider theme={myTheme}>
    <ResetPassword />
  </MuiThemeProvider>
  , document.getElementById('reset-root'))
