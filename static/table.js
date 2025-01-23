function toggleRows() {
    var rows = document.querySelectorAll('.hidden-rows');
    var button = document.getElementById('toggleButton');
    rows.forEach(row => {
        if (row.style.display === 'none') {
            row.style.display = 'table-row';
            button.textContent = 'Show Less';
        } else {
            row.style.display = 'none';
            button.textContent = 'Show More';
        }
    });
}