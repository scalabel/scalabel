import { MuiThemeProvider } from '@material-ui/core/styles'
import { Router } from '@reach/router'
import React from 'react'
import { render } from 'react-dom'
import AccessHome from '../components/access_control/accessHome'
import PermissionList from '../components/access_control/permissionList'
import { myTheme } from '../styles/theme'

render(
  <MuiThemeProvider theme={myTheme}>
    <Router>
      <AccessHome path='access'>
        <PermissionList path='permissions' />
      </AccessHome>
    </Router>
  </MuiThemeProvider>
  , document.getElementById('root'))
