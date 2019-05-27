import {initSession} from '../common/session_init';
import ReactDOM from 'react-dom';
initSession();
// $FlowFixMe
ReactDOM.render(document.getElementById('labeling-interface'));
