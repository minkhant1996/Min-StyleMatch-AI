def render_face_data_html(face_data: dict) -> str:
    html = "<table style='border-collapse: collapse; width: 100%; max-width: 600px;'>"
    html += "<tr><th style='text-align:left; padding: 4px;'>Feature</th><th style='text-align:left; padding: 4px;'>Color</th><th style='text-align:left; padding: 4px;'>Hex Code</th></tr>"

    if not face_data:
        html += "<tr><td colspan='3' style='padding: 6px; text-align: center;'>No face analysis data available.</td></tr>"
    else:
        for key, value in face_data.items():
            html += f"""
            <tr>
                <td style='padding: 4px;'>{key}</td>
                <td style='padding: 4px;'>
                    <div style='width: 40px; height: 20px; background-color: {value}; border: 1px solid #ccc;'></div>
                </td>
                <td style='padding: 4px;'>{value}</td>
            </tr>
            """

    html += "</table>"
    return html
