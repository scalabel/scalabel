import { MuiThemeProvider } from '@material-ui/core/styles'
import React from 'react'
import ReactDOM from 'react-dom'
import Login from '../components/login/login'
import { myTheme } from '../styles/theme'

ReactDOM.render(
  <MuiThemeProvider theme={myTheme}>
    <Login />
  </MuiThemeProvider>
  , document.getElementById('login-root'))
