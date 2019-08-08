import { MuiThemeProvider } from '@material-ui/core/styles'
import React from 'react'
import ReactDOM from 'react-dom'
import Dashboard from '../components/dashboard_worker'
import { myTheme } from '../styles/theme'

ReactDOM.render(
        <MuiThemeProvider theme={myTheme}>
          <Dashboard/>
        </MuiThemeProvider>, document.getElementById('worker'))
