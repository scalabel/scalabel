import { MuiThemeProvider } from '@material-ui/core/styles'
import React from 'react'
import ReactDOM from 'react-dom'
import Create from '../components/create_project'
import { myTheme } from '../styles/theme'

ReactDOM.render(
  <MuiThemeProvider theme={myTheme}>
    <Create />
  </MuiThemeProvider>
  , document.getElementById('create'))
