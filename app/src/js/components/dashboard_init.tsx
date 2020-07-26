import { MuiThemeProvider } from '@material-ui/core/styles'
import React from 'react'
import ReactDOM from 'react-dom'
import { getAuth } from '../common/service'
import { Endpoint } from '../const/connection'
import { myTheme } from '../styles/theme'
import Dashboard, { DashboardContents } from './dashboard'

/**
 * This function post requests to backend to retrieve dashboard contents
 */
export function initDashboard (vendor?: boolean) {
  let dashboardContents: DashboardContents
  const xhr = new XMLHttpRequest()
  xhr.onreadystatechange = () => {
    if (xhr.readyState === 4 && xhr.status === 200) {
      dashboardContents = JSON.parse(xhr.responseText)
      ReactDOM.render(
        <MuiThemeProvider theme={myTheme}>
          <Dashboard dashboardContents={dashboardContents}
            vendor={vendor} />
        </MuiThemeProvider>
        , document.getElementById(vendor ? 'vendor-root'
          : 'dashboard-root'))
    }
  }
  // Get params from url path.
  const searchParams = new URLSearchParams(window.location.search)
  const projectName = searchParams.get('project_name')
  // Send the request to the back end
  const request = JSON.stringify({
    name: projectName
  })

  xhr.open('POST', Endpoint.DASHBOARD)
  xhr.setRequestHeader('Content-Type', 'application/json')
  const auth = getAuth()
  if (auth) {
    xhr.setRequestHeader('Authorization', auth)
  }
  xhr.send(request)
}
