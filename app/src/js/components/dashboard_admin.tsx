import React from 'react';
import theme from '../styles/theme';
import classNames from 'classnames';
import Paper from '@material-ui/core/Paper';
import Drawer from '@material-ui/core/Drawer';
import AppBar from '@material-ui/core/AppBar';
import Toolbar from '@material-ui/core/Toolbar';
import Typography from '@material-ui/core/Typography';
import IconButton from '@material-ui/core/IconButton';
import {withStyles} from '@material-ui/core/styles';
import CssBaseline from '@material-ui/core/CssBaseline';
import {MuiThemeProvider} from '@material-ui/core/styles';
import {
    getProjects,
    getUsers,
    goCreate,
    logout,
    toProject
} from '../common/service';
import {
    dashboardStyles,
    tableCellStyles,
    tableStyles
} from '../styles/dashboard';
// lists
import List from '@material-ui/core/List';
import Divider from '@material-ui/core/Divider';
import ListItem from '@material-ui/core/ListItem';
import ListItemIcon from '@material-ui/core/ListItemIcon';
import ListItemText from '@material-ui/core/ListItemText';
// icons
import SvgIcon from '@material-ui/core/SvgIcon';
import CreateIcon from '@material-ui/icons/Create';
// table
import Table from '@material-ui/core/Table';
import TableRow from '@material-ui/core/TableRow';
import TableBody from '@material-ui/core/TableBody';
import TableCell from '@material-ui/core/TableCell';
import TableHead from '@material-ui/core/TableHead';

/* Retrieve data from backend */
const usersToExpress = getUsers();
const projectsToExpress = getProjects();

/* Theme for dashboard, set main color as grey */
const myTheme = theme({ palette: { primary: {main: '#616161'} }});

/* Sidebar: mainList */
export const mainListItems = (
  <div>
    <ListItem button onClick={goCreate}>
      <ListItemIcon>
        <CreateIcon />
      </ListItemIcon>
      <ListItemText primary='Create new project' />
    </ListItem>
  </div>
);

/**
 * This is Dashboard component that displays
 * the everything post in the dashboard.
 * @param {Object} props
 * @return component
 */
function Dashboard(props: {
  /** styles of Dashboard */
  classes: any; }) {
  const {classes} = props;
    /**
     * render function
     * @return component
     */
  return (
    <div className={classes.root}>
        <CssBaseline/>
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
                    Scalabel Admin Dashboard
                </Typography>
                <IconButton className={classes.logout} onClick={logout}>
                  <SvgIcon >
                      <path d='M10.09 15.59L11.5 17l5-5-5-5-1.41 1.41L12.67
                    11H3v2h9.67l-2.58 2.59zM19 3H5c-1.11 0-2 .9-2
                    2v4h2V5h14v14H5v-4H3v4c0 1.1.89 2 2 2h14c1.1 0 2-.9
                    2-2V5c0-1.1-.9-2-2-2z' fill='#ffffff'/>
                  </SvgIcon>
                </IconButton>
            </Toolbar>
        </AppBar>
        <Drawer
            variant='permanent'
            classes={{
                paper: classNames(classes.drawerPaper)
            }}
        >
            <div className={classes.toolbarIcon}/>
            <Divider/>
            <List>{mainListItems}</List>
            <Divider/>
        </Drawer>
        <main className={classes.content}>
            <div className={classes.appBarSpacer}/>
            <Typography variant='h6' gutterBottom component='h2'>
                Projects
            </Typography>
            <Typography component='div' className={classes.chartContainer}>
                <ProjectTableDisplay classes = {tableStyles}/>
            </Typography>
            <div><br/></div>
            <Typography variant='h6' gutterBottom component='h2'>
                Users Lists
            </Typography>
            <Typography component='div' className={classes.chartContainer}>
                <WorkersTableDisplay classes = {tableStyles}/>
            </Typography>
        </main>
    </div>);
}

const DashboardTableCell = withStyles(tableCellStyles)(TableCell);

/**
 * This is projectTable component that displays
 * all the information about projects
 * @param {object} Props
 * @return component
 */
const ProjectTable = function(Props: {
  /** styles of ProjectTable */
  classes: any; }) {
  const {classes} = Props;
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
          {projectsToExpress.map((row, i) => (
            <TableRow className={classes.row} key={i}>
              <DashboardTableCell className={'align'} onClick={() => {
                toProject(row);
                }} component='th' scope='row'>
                {row}
              </DashboardTableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Paper>
  );
};

const ProjectTableDisplay = withStyles(tableStyles)(ProjectTable);

/**
 * This is WorkersTable component that displays
 * all the information about workers
 * @param props
 * @return component
 */
const WorkersTable = function(props: {
  /** styles of ProjectTable */
  classes: any; }) {
  const {classes} = props;
  return (
    <Paper className={classes.root}>
      <Table className={classes.table}>
        <TableHead>
          <TableRow>
            <DashboardTableCell>Email</DashboardTableCell>
            <DashboardTableCell align='right'>Group</DashboardTableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {usersToExpress.map((row, i) => (
            <TableRow className={classes.row} key={i}>
              <DashboardTableCell component='th' scope='row'>
                {row.Email}
              </DashboardTableCell>
              <DashboardTableCell align='right'>{row.Group}</DashboardTableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Paper>
  );
};

const WorkersTableDisplay = withStyles(tableStyles)(WorkersTable);

/** export Dashboard */
export default withStyles(dashboardStyles)(Dashboard);
