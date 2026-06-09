let ele = document.getElementById('h2_style');
console.log("查看一下:", ele);
ele.onclick = function () {
    this.style.color = "red";
    console.log("已经点击");
}