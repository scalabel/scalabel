import { List, ListItemIcon } from '@material-ui/core'
import Divider from '@material-ui/core/Divider'
import ListItem from '@material-ui/core/ListItem'
import ListItemText from '@material-ui/core/ListItemText'
import { withStyles } from '@material-ui/core/styles'
import Typography from '@material-ui/core/Typography'
import CreateIcon from '@material-ui/icons/Create'
import React from 'react'
import { getProjects, goCreate, requestData } from '../common/service'
import { dashboardStyles } from '../styles/dashboard'
import DashboardHeader from './dashboard_header'
import DataTable from './dashboard_table'
import { DashboardClassType, User } from './dashboard_worker'
import DividedPage from './divided_page'

interface AdminClassType extends DashboardClassType {
  /** admin root */
  adminRoot: string
}

/**
 * This function post request to backend to retrieve users' information
 * @return {function} users
 */
export function getUsers (): User[] {
  return requestData('./postUsers', 'get', false)
}

/**
 * This is Dashboard component that displays
 * the admin dashboard.
 * @param {object} props
 * @return component
 */
function Dashboard (props: {
  /** style of admin dashboard */
  classes: AdminClassType
}) {
  const { classes } = props
  const headerContent = (
    <DashboardHeader admin />
  )
  let usersToDisplay: User[]
  const users = getUsers()
  if (users.length > 0) {
    usersToDisplay = users
  } else {
    usersToDisplay = [{
      email: 'no user data available', group: '0',
      id: '', projects: [''], refreshToken: ''
    }]
  }
  const sidebarContent = (
    <List>
      <ListItem button
        onClick={goCreate}
      >
        <ListItemIcon>
          <CreateIcon />
        </ListItemIcon>
        <ListItemText primary={'Create new project'} />
      </ListItem>

    </List>
  )
  const mainContent = (
    <div className={classes.adminRoot}>
      <Typography variant='h6' component='h2'
        className={classes.labelText}>
        Projects
            </Typography>
      <Divider />
      <DataTable dataList={getProjects()}
        headers={[{ header: 'Project', align: 'left' }]}
      />
      <Typography variant='h6' component='h2'
        className={classes.labelText}>
        Users
            </Typography>
      <Divider />
      <DataTable dataList={usersToDisplay}
        headers={[{ header: 'Email', align: 'left' },
        { header: 'Group', align: 'right' }]}
      />
    </div>
  )
  /**
   * render function
   * @return component
   */
  return (
    <DividedPage children={{
      headerContent,
      sidebarContent,
      mainContent
    }}
    />
  )
}

/** export Dashboard */
export default withStyles(dashboardStyles, { withTheme: true })(Dashboard)
