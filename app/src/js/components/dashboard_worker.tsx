// tslint:disable:no-any
// TODO: remove the disable tag
import AppBar from '@material-ui/core/AppBar'
import CssBaseline from '@material-ui/core/CssBaseline'
import IconButton from '@material-ui/core/IconButton'
import Paper from '@material-ui/core/Paper'
import { MuiThemeProvider, withStyles } from '@material-ui/core/styles'
// icons
import SvgIcon from '@material-ui/core/SvgIcon'
// table
import Table from '@material-ui/core/Table'
import TableBody from '@material-ui/core/TableBody'
import TableCell from '@material-ui/core/TableCell'
import TableHead from '@material-ui/core/TableHead'
import TableRow from '@material-ui/core/TableRow'
import Toolbar from '@material-ui/core/Toolbar'
import Typography from '@material-ui/core/Typography'
import classNames from 'classnames'
import React from 'react'
import { getProjects, logout, toProject } from '../common/service'
import {
  dashboardStyles,
  tableCellStyles,
  tableStyles
} from '../styles/dashboard'
import theme from '../styles/theme'

/* Retrieve data from backend */
const projectsToExpress = getProjects()

/* Theme for dashboard, set main color as grey */
const myTheme = theme({ palette: { primary: { main: '#616161' } } })

/**
 * This is Dashboard component that displays
 * the everything post in the dashboard.
 * @param {object} props
 * @return component
 */
function Dashboard (props: any) {
  const { classes } = props
    /**
     * render function
     * @return component
     */
  return (
      <div className={classes.root}>
        <CssBaseline />
          <AppBar
            position='absolute'
            className={classNames(classes.appBar)}
          >
            <Toolbar className={classes.toolbar}>
              <Typography
                component='h1'
                variant='h6'
                color='inherit'
                noWrap
                className={classes.title}
              >
                Scalabel Worker Dashboard
              </Typography>
                <IconButton
                    className={classNames(classes.logout)}
                    onClick={logout}
                >
                  <SvgIcon >
                    <path d='M10.09 15.59L11.5 17l5-5-5-5-1.41 1.41L12.67
                    11H3v2h9.67l-2.58 2.59zM19 3H5c-1.11 0-2 .9-2
                    2v4h2V5h14v14H5v-4H3v4c0 1.1.89 2 2 2h14c1.1 0 2-.9
                    2-2V5c0-1.1-.9-2-2-2z' fill='#ffffff'/>
                  </SvgIcon>
                </IconButton>
            </Toolbar>
          </AppBar>
        <main className={classes.content}>
          <div className={classes.appBarSpacer} />
          <Typography variant='h6' gutterBottom component='h2'>
            Projects
          </Typography>
          <Typography component='div' className={classes.chartContainer}>
            <DashboardTable classes = {tableStyles}/>
          </Typography>
        </main>
      </div>
  )
}
const DashboardTableCell = withStyles(tableCellStyles)(TableCell)

/**
 * This is projectTable component that displays
 * all the information about projects
 * @param {object} props
 * @return component
 */
const ProjectTable = (props: {
  /** styles of Project Table */
  classes: any; }) => {
  const { classes } = props
  return (
    <Paper className={classes.root}>
      <Table className={classes.table}>
        <MuiThemeProvider theme={myTheme}>
          <TableHead >
            <TableRow>
              <DashboardTableCell>Projects</DashboardTableCell>
            </TableRow>
          </TableHead>
        </MuiThemeProvider>
        <TableBody>
          {projectsToExpress.map((row: any, i: any) => (
            <TableRow className={classes.row} key={i}>
              <DashboardTableCell onClick={() => {
                toProject(row)
              }} component='th' scope='row'>
                {row}
              </DashboardTableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Paper>
  )
}
const DashboardTable = withStyles(tableStyles)(ProjectTable)

/** export Dashboard */
export default withStyles(dashboardStyles)(Dashboard)
