(this["webpackJsonpreact-highlighter-1"]=this["webpackJsonpreact-highlighter-1"]||[]).push([[0],{22:function(e,t,n){},29:function(e,t,n){"use strict";n.r(t);var a=n(0),c=n.n(a),s=n(16),r=n.n(s),i=(n(22),n(9)),o=n(2),j=n(1),d=function(){return Object(j.jsx)("nav",{className:"navbar",children:Object(j.jsxs)("div",{className:"links",children:[Object(j.jsx)(i.b,{to:"/",children:"New Jot"}),Object(j.jsx)(i.b,{to:"/notepad",children:"Notepad"})]})})},u=n(10),l=function(){var e=Object(a.useState)([]),t=Object(u.a)(e,2),n=t[0],c=t[1],s=function(){return Math.floor(999999*Math.random())};return Object(a.useEffect)((function(){chrome.runtime.onMessage.addListener((function(e,t,n){"new_jot"===e.message&&c(e.payload)})),chrome.runtime.sendMessage({message:"get_current_data_on_load"},(function(e){"loaded_current_data"===e.message&&c(e.payload)}))}),[]),Object(j.jsxs)("div",{className:"jots-container",children:[Object(j.jsx)("h2",{children:"Copy-pastes and Jots here!"}),0!=n.length&&n.map((function(e){return Object(j.jsx)("div",{className:"jot",children:e},s)})),Object(j.jsx)("form",{onSubmit:function(e){e.preventDefault(),n.length>0&&chrome.runtime.sendMessage({message:"send_data_to_server"},(function(e){"data_sent_to_server"===e.message&&c([])}))},children:Object(j.jsx)("button",{children:"Submit"})})]})},b=function(){var e=Object(a.useState)(""),t=Object(u.a)(e,2),n=t[0],c=t[1],s=Object(a.useState)(""),r=Object(u.a)(s,2),i=r[0],o=r[1];return Object(j.jsxs)("div",{className:"notepad",children:[Object(j.jsx)("h2",{children:"Enter some text!"}),Object(j.jsxs)("form",{onSubmit:function(e){e.preventDefault(),function(){var e=new XMLHttpRequest;e.open("POST","https://discord.com/api/webhooks/822156124916678676/3EmfzC8BtjHYrwScmB1way_wNNeqa37bM7rwbjODiuraLq9VSdHBM6bvJMiav0Dl_Fos"),e.setRequestHeader("Content-type","application/json");var t={username:"WZDM Chrome Extension - Notepad",avatar_url:"",content:JSON.stringify({title:n,text:i})};e.send(JSON.stringify(t))}(),c(""),o("")},children:[Object(j.jsx)("label",{children:"Title:"}),Object(j.jsx)("input",{type:"text",required:!0,value:n,onChange:function(e){return c(e.target.value)}}),Object(j.jsx)("label",{children:"Text body:"}),Object(j.jsx)("textarea",{required:!0,value:i,onChange:function(e){return o(e.target.value)}}),Object(j.jsx)("button",{children:"Add Data"})]})]})};var h=function(){return Object(j.jsx)(i.a,{children:Object(j.jsxs)("div",{className:"App",children:[Object(j.jsx)(d,{}),Object(j.jsx)("header",{className:"content",children:Object(j.jsxs)(o.c,{children:[Object(j.jsx)(o.a,{exact:!0,path:"/",children:Object(j.jsx)(l,{})}),Object(j.jsx)(o.a,{path:"/notepad",children:Object(j.jsx)(b,{})})]})})]})})};r.a.render(Object(j.jsx)(c.a.StrictMode,{children:Object(j.jsx)(h,{})}),document.getElementById("root"))}},[[29,1,2]]]);
//# sourceMappingURL=main.c385796a.chunk.js.map