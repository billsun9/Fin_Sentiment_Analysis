import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';
import Navbar from './components/Navbar';
import Jots from './components/Jots';
import Notepad from './components/Notepad';
function App() {
  return (
    <Router>
      <div className="App">
        <Navbar />
        <header className="content">
          <Switch>
            <Route exact path="/">
              <Jots />
            </Route>
            <Route path="/notepad">
              <Notepad />
            </Route>
          </Switch>
        </header>
      </div>
    </Router>
    
  );
}

export default App;
